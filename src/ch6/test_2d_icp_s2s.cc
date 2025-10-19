//
// Created by xiang on 2022/3/15.
//
#include <opencv2/highgui.hpp>
#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "ch6/icp_2d.h"
#include "ch6/lidar_2d_utils.h"
#include "common/io_utils.h"

DEFINE_string(bag_path, "./dataset/sad/2dmapping/floor3.bag", "数据包路径");
DEFINE_string(method, "point2point", "2d icp方法：[point2point,point2point_g2o,point2plane,point2plane_g2o]");
DEFINE_int32(sample_interval, 1, "采样间隔");

/// 测试从rosbag中读取2d scan并plot的结果
/// 通过选择method来确定使用点到点或点到面的ICP
int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::RosbagIO rosbag_io(FLAGS_bag_path);
    Scan2d::Ptr last_scan = nullptr, current_scan = nullptr;

    /// 我们将上一个scan与当前scan进行配准
    rosbag_io
        .AddScan2DHandle("/pavo_scan_bottom",
                         [&](Scan2d::Ptr scan) {
                             {
                                 static int count = 0;
                                 if (count++ % FLAGS_sample_interval != 0) {
                                     return true;
                                 }
                             }
                             current_scan = scan;

                             if (last_scan == nullptr) {
                                 last_scan = current_scan;
                                 return true;
                             }

                             sad::Icp2d icp;
                             icp.SetTarget(last_scan);
                             icp.SetSource(current_scan);

                             SE2 pose;
                             if (FLAGS_method == "point2point") {
                                 icp.AlignGaussNewton(pose);
                             } else if (FLAGS_method == "point2point_g2o") {
                                 icp.AlignG2O(pose);
                             } else if (FLAGS_method == "point2plane") {
                                 icp.AlignGaussNewtonPoint2Plane(pose);
                             } else if (FLAGS_method == "point2plane_g2o") {
                                 icp.AlignG2OPoint2Plane(pose);
                             } else {
                                 spdlog::error("unknown method: {}", FLAGS_method);
                                 return false;
                             }

                             //  正常效果应该是能通过黑点看出相对位移，且上一帧和当前帧的扫描数据重合度较高

                             cv::Mat image;
                             sad::Visualize2DScan(last_scan, SE2(), image, Vec3b(255, 0, 0));    // target是蓝的
                             sad::Visualize2DScan(current_scan, pose, image, Vec3b(0, 0, 255));  // source是红的
                             cv::imshow("scan", image);
                             cv::waitKey(10 * FLAGS_sample_interval);

                             last_scan = current_scan;
                             return true;
                         })
        .Go();

    return 0;
}
