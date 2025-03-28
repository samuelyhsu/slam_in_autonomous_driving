//
// Created by xiang on 2022/7/18.
//

#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "ch7/loam-like/feature_extraction.h"
#include "common/io_utils.h"

#include "common/point_cloud_utils.h"
#include "common/timer/timer.h"

/// 这里需要vlp16的数据，用wxb的
DEFINE_string(bag_path, "./dataset/sad/wxb/test1.bag", "path to wxb bag");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    // 测试角点和平面点的提取
    sad::FeatureExtraction feature_extraction;

    system("rm -rf ./data/ch7/*.pcd");

    sad::RosbagIO bag_io(fLS::FLAGS_bag_path);
    bag_io
        .AddVelodyneHandle("/velodyne_packets_1",
                           [&](sad::FullCloudPtr cloud) -> bool {
                               sad::CloudPtr pcd_corner(new sad::PointCloudType), pcd_surf(new sad::PointCloudType);
                               sad::common::Timer::Evaluate(
                                   [&]() { feature_extraction.Extract(cloud, pcd_corner, pcd_surf); },
                                   "Feature Extraction");
                               //    pcd_corner->size()
                               //                 << ", surf: " << pcd_surf->size();
                               spdlog::info("original pts: {}, corners: {}, surf: {}", cloud->size(),
                                            pcd_corner->size(), pcd_surf->size());
                               sad::SaveCloudToFile("./data/ch7/corner.pcd", *pcd_corner);
                               sad::SaveCloudToFile("./data/ch7/surf.pcd", *pcd_surf);
                               return true;
                           })
        .Go();

    sad::common::Timer::PrintAll();
    spdlog::info("done.");

    return 0;
}
