//
// Created by xiang on 2022/1/21.
//

#include "ch3/static_imu_init.h"
#include "ch3/utm_convert.h"
#include "ch4/gins_pre_integ.h"
#include "common/io_utils.h"
#include "tools/ui/pangolin_window.h"

#include <fstream>
#include <iomanip>
#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

/**
 * 运行由预积分构成的GINS系统
 */
DEFINE_string(txt_path, "./data/ch3/10.txt", "数据文件路径");
DEFINE_double(antenna_angle, 12.06, "RTK天线安装偏角（角度）");
DEFINE_double(antenna_pox_x, -0.17, "RTK天线安装偏移X");
DEFINE_double(antenna_pox_y, -0.20, "RTK天线安装偏移Y");
DEFINE_bool(with_ui, true, "是否显示图形界面");
DEFINE_bool(debug, false, "是否打印调试信息");
DEFINE_bool(with_odom, false, "是否加入轮速计信息");
DEFINE_bool(use_gnss_heading, false, "use gnss heading");
DEFINE_double(observe_interval, 0, "gnss observe interval in seconds");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (fLS::FLAGS_txt_path.empty()) {
        return -1;
    }

    // 初始化器
    sad::StaticIMUInit imu_init;  // 使用默认配置

    sad::TxtIO io(fLS::FLAGS_txt_path);
    Vec2d antenna_pos(fLD::FLAGS_antenna_pox_x, fLD::FLAGS_antenna_pox_y);

    auto save_vec3 = [](std::ofstream& fout, const Vec3d& v) { fout << v[0] << " " << v[1] << " " << v[2] << " "; };
    auto save_quat = [](std::ofstream& fout, const Quatd& q) {
        fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
    };

    auto save_result = [&save_vec3, &save_quat](std::ofstream& fout, const sad::NavStated& save_state) {
        fout << std::setprecision(18) << save_state.timestamp_ << " " << std::setprecision(9);
        save_vec3(fout, save_state.p_);
        save_quat(fout, save_state.R_.unit_quaternion());
        save_vec3(fout, save_state.v_);
        save_vec3(fout, save_state.bg_);
        save_vec3(fout, save_state.ba_);
        fout << std::endl;
    };

    std::ofstream fout("./data/ch4/gins_preintg.txt");
    bool imu_inited = false, gnss_inited = false;

    sad::GinsPreInteg::Options gins_options;
    gins_options.verbose_ = FLAGS_debug;
    sad::GinsPreInteg gins(gins_options);

    bool first_gnss_set = false;
    Vec3d origin = Vec3d::Zero();

    std::shared_ptr<sad::ui::PangolinWindow> ui = nullptr;
    if (FLAGS_with_ui) {
        ui = std::make_shared<sad::ui::PangolinWindow>();
        ui->Init();
    }

    /// 设置各类回调函数
    io.SetIMUProcessFunc([&](const sad::IMU& imu) {
          /// IMU 处理函数
          if (!imu_init.InitSuccess()) {
              imu_init.AddIMU(imu);
              return;
          }

          /// 需要IMU初始化
          if (!imu_inited) {
              // 读取初始零偏，设置GINS
              sad::GinsPreInteg::Options options;
              options.preinteg_options_.init_bg_ = imu_init.GetInitBg();
              options.preinteg_options_.init_ba_ = imu_init.GetInitBa();
              options.gravity_ = imu_init.GetGravity();
              gins.SetOptions(options);
              imu_inited = true;
              return;
          }

          if (!gnss_inited) {
              /// 等待有效的RTK数据
              return;
          }

          /// GNSS 也接收到之后，再开始进行预测
          gins.AddImu(imu);

          auto state = gins.GetState();
          save_result(fout, state);
          if (ui) {
              ui->UpdateNavState(state);
              //   usleep(5e2);
          }
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

            static double prev_time;
            if (gnss_convert.unix_time_ - prev_time < FLAGS_observe_interval) {
                return;
            }
            prev_time = gnss_convert.unix_time_;
            if (!FLAGS_use_gnss_heading) {
                gnss_convert.heading_valid_ = false;
            }

            /// 去掉原点
            if (!first_gnss_set) {
                origin = gnss_convert.utm_pose_.translation();
                first_gnss_set = true;
            }
            gnss_convert.utm_pose_.translation() -= origin;

            gins.AddGnss(gnss_convert);

            auto state = gins.GetState();
            save_result(fout, state);
            if (ui) {
                ui->UpdateNavState(state);
                // usleep(1e3);
            }
            gnss_inited = true;
        })
        .SetOdomProcessFunc([&](const sad::Odom& odom) {
            imu_init.AddOdom(odom);

            if (FLAGS_with_odom && imu_inited && gnss_inited) {
                gins.AddOdom(odom);
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
