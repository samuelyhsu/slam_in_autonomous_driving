//
// Created by xiang on 2021/11/11.
//

#include "ch3/eskf.hpp"
#include "ch3/static_imu_init.h"
#include "common/io_utils.h"
#include "tools/ui/pangolin_window.h"
#include "utm_convert.h"

#include <fstream>
#include <iomanip>
#include "common/math_utils.h"
#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

DEFINE_string(txt_path, "./data/ch3/10.txt", "数据文件路径");
DEFINE_double(antenna_angle, 12.06, "RTK天线安装偏角（角度）");
DEFINE_double(antenna_pox_x, -0.17, "RTK天线安装偏移X");
DEFINE_double(antenna_pox_y, -0.20, "RTK天线安装偏移Y");
DEFINE_bool(with_ui, true, "是否显示图形界面");
DEFINE_bool(with_odom, false, "是否加入轮速计信息");
DEFINE_bool(use_gnss_heading, false, "use gnss heading");
DEFINE_double(observe_interval, 0, "gnss observe interval in seconds");
DEFINE_double(ui_ms_delay, 0, "ui update delay in ms");

/**
 * 本程序演示使用RTK+IMU进行组合导航
 */
int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (fLS::FLAGS_txt_path.empty()) {
        return -1;
    }

    // 初始化器
    sad::StaticIMUInit imu_init;  // 使用默认配置
    sad::ESKFD eskf;

    sad::TxtIO io(FLAGS_txt_path);
    Vec2d antenna_pos(FLAGS_antenna_pox_x, FLAGS_antenna_pox_y);

    auto save_vec3 = [](std::ofstream& fout, const Vec3d& v) { fout << v[0] << " " << v[1] << " " << v[2] << " "; };
    auto save_quat = [](std::ofstream& fout, const Quatd& q) {
        fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
    };

    auto save_result = [&save_vec3, &save_quat](std::ofstream& fout, const sad::NavStated& save_state,
                                                const Vec3d& gravity) {
        fout << std::setprecision(18) << save_state.timestamp_ << " " << std::setprecision(9);
        save_vec3(fout, save_state.p_);
        save_quat(fout, save_state.R_.unit_quaternion());
        save_vec3(fout, save_state.v_);
        save_vec3(fout, save_state.bg_);
        save_vec3(fout, save_state.ba_);
        save_vec3(fout, gravity);
        fout << std::endl;
    };

    std::ofstream fout("./data/ch3/gins.txt");
    bool imu_inited = false, gnss_inited = false;

    std::shared_ptr<sad::ui::PangolinWindow> ui = nullptr;
    if (FLAGS_with_ui) {
        ui = std::make_shared<sad::ui::PangolinWindow>();
        ui->Init();
    }

    /// 设置各类回调函数
    bool first_gnss_set = false;
    Vec3d origin = Vec3d::Zero();

    static double prev_gnss_time;

    io.SetIMUProcessFunc([&](const sad::IMU& imu) {
          /// IMU 处理函数
          if (!imu_init.InitSuccess()) {
              imu_init.AddIMU(imu);
              return;
          }

          /// 需要IMU初始化
          if (!imu_inited) {
              // 读取初始零偏，设置ESKF
              sad::ESKFD::Options options;
              // 噪声由初始化器估计
              options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
              options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
              //   options.bias_gyro_var_ = 1e-6;
              //   options.bias_acce_var_ = 1e-4;
              //   options.gnss_ang_noise_ = 5 * sad::math::kDEG2RAD;
              eskf.SetInitialConditions(options, imu_init.GetInitBg(), imu_init.GetInitBa(), imu_init.GetGravity());
              imu_inited = true;
              return;
          }

          if (!gnss_inited) {
              /// 等待有效的RTK数据
              return;
          }

          /// GNSS 也接收到之后，再开始进行预测
          eskf.Predict(imu);

          /// predict就会更新ESKF，所以此时就可以发送数据
          auto state = eskf.GetNominalState();
          auto gravity = eskf.GetGravity();

          if (ui) {
              ui->UpdateNavState(state);
          }

          /// 记录数据以供绘图
          save_result(fout, state, gravity);
      })
        .SetGNSSProcessFunc([&](const sad::GNSS& gnss) {
            /// GNSS 处理函数
            if (!imu_inited) {
                return;
            }

            sad::GNSS gnss_convert = gnss;
            if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos, FLAGS_antenna_angle)) {
                return;
            }

#if 1
            static double prev_time;
            if (gnss_convert.unix_time_ - prev_time < FLAGS_observe_interval) {
                return;
            }
            prev_time = gnss_convert.unix_time_;
            if (!FLAGS_use_gnss_heading) {
                gnss_convert.heading_valid_ = false;
            }

#endif

            prev_gnss_time = prev_time;

            /// 去掉原点
            if (!first_gnss_set) {
                origin = gnss_convert.utm_pose_.translation();
                first_gnss_set = true;
            }
            gnss_convert.utm_pose_.translation() -= origin;

            // 要求RTK heading有效，才能合入ESKF
            eskf.ObserveGps(gnss_convert);

            auto state = eskf.GetNominalState();
            auto gravity = eskf.GetGravity();
            if (ui) {
                ui->UpdateNavState(state);
            }
            save_result(fout, state, gravity);

            gnss_inited = true;
            usleep(FLAGS_ui_ms_delay * 1e3);
        })
        .SetOdomProcessFunc([&](const sad::Odom& odom) {
            /// Odom 处理函数，本章Odom只给初始化使用
            imu_init.AddOdom(odom);
            if (FLAGS_with_odom && imu_inited && gnss_inited) {
                eskf.ObserveWheelSpeed(odom);
            }
        })
        .Go();

    while (ui && !ui->ShouldQuit()) {
        usleep(1e5);
    }
    if (ui) {
        ui->Quit();
    }
    return 0;
}
