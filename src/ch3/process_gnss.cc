//
// Created by xiang on 2022/1/4.
//

#include <iomanip>
#include <memory>
#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#include "common/gnss.h"
#include "common/io_utils.h"
#include "gflags/gflags.h"
#include "tools/ui/pangolin_window.h"
#include "utm_convert.h"

DEFINE_string(txt_path, "./data/ch3/10.txt", "数据文件路径");

// 以下参数仅针对本书提供的数据
DEFINE_double(antenna_angle, 12.06, "RTK天线安装偏角（角度）");
DEFINE_double(antenna_pox_x, -0.17, "RTK天线安装偏移X");
DEFINE_double(antenna_pox_y, -0.20, "RTK天线安装偏移Y");
DEFINE_bool(with_ui, true, "是否显示图形界面");
DEFINE_double(observe_interval, 0, "gnss observe interval in seconds");
DEFINE_bool(only_heading_valid, false, "only use heading valid data");
DEFINE_double(ui_ms_delay, 0, "ui update delay in ms");

/**
 * 本程序演示如何处理GNSS数据
 * 我们将GNSS原始读数处理成能够进行后续处理的6自由度Pose
 * 需要处理UTM转换、RTK天线外参、坐标系转换三个步骤
 *
 * 我们将结果保存在文件中，然后用python脚本进行可视化
 */

int main(int argc, char** argv) {
    spdlog::info("{}", fmt::join(argv, argv + argc, " "));
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (fLS::FLAGS_txt_path.empty()) {
        return -1;
    }

    sad::TxtIO io(fLS::FLAGS_txt_path);

    std::ofstream fout("./data/ch3/gnss_output.txt");
    Vec2d antenna_pos(FLAGS_antenna_pox_x, FLAGS_antenna_pox_y);

    auto save_result = [](std::ofstream& fout, double timestamp, const SE3& pose) {
        auto save_vec3 = [](std::ofstream& fout, const Vec3d& v) { fout << v[0] << " " << v[1] << " " << v[2] << " "; };
        auto save_quat = [](std::ofstream& fout, const Quatd& q) {
            fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
        };

        fout << std::setprecision(18) << timestamp << " " << std::setprecision(9);
        save_vec3(fout, pose.translation());
        save_quat(fout, pose.unit_quaternion());
        fout << std::endl;
    };

    std::shared_ptr<sad::ui::PangolinWindow> ui = nullptr;
    if (FLAGS_with_ui) {
        ui = std::make_shared<sad::ui::PangolinWindow>();
        ui->Init();
    }

    bool first_gnss_set = false;
    Vec3d origin = Vec3d::Zero();
    io.SetGNSSProcessFunc([&](const sad::GNSS& gnss) {
          sad::GNSS gnss_out = gnss;
          if (sad::ConvertGps2UTM(gnss_out, antenna_pos, FLAGS_antenna_angle)) {
#if 1
              if (FLAGS_only_heading_valid) {
                  if (!gnss_out.heading_valid_) {
                      return;
                  }
              }
              static double prev_time;
              if (gnss_out.unix_time_ - prev_time < FLAGS_observe_interval) {
                  return;
              }
              prev_time = gnss_out.unix_time_;
#endif
              if (!first_gnss_set) {
                  origin = gnss_out.utm_pose_.translation();
                  first_gnss_set = true;
              }

              /// 减掉一个原点
              gnss_out.utm_pose_.translation() -= origin;

              save_result(fout, gnss_out.unix_time_, gnss_out.utm_pose_);

              if (ui) {
                  ui->UpdateNavState(
                      sad::NavStated(gnss_out.unix_time_, gnss_out.utm_pose_.so3(), gnss_out.utm_pose_.translation()));
                  usleep(FLAGS_ui_ms_delay * 1e3);
              }
          }
      }).Go();

    if (ui) {
        while (!ui->ShouldQuit()) {
            usleep(1e5);
        }
        ui->Quit();
    }

    return 0;
}
