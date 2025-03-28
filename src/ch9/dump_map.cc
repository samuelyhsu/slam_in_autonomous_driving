//
// Created by xiang on 22-12-7.
//

#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

DEFINE_double(voxel_size, 0.1, "导出地图分辨率");
DEFINE_string(pose_source, "lidar", "使用的pose来源:lidar/rtk/opti1/opti2");
DEFINE_string(dump_to, "./data/ch9/", "导出的目标路径");

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "common/point_cloud_utils.h"
#include "keyframe.h"

/**
 * 将keyframes.txt中的地图和点云合并为一个pcd
 */

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    using namespace sad;
    std::map<IdType, KFPtr> keyframes;
    if (!LoadKeyFrames("./data/ch9/keyframes.txt", keyframes)) {
        spdlog::error("failed to load keyframes.txt");
        return -1;
    }

    if (keyframes.empty()) {
        spdlog::info("keyframes are empty");
        return 0;
    }

    // dump kf cloud and merge
    spdlog::info("merging");
    CloudPtr global_cloud(new PointCloudType);

    pcl::VoxelGrid<PointType> voxel_grid_filter;
    float resolution = FLAGS_voxel_size;
    voxel_grid_filter.setLeafSize(resolution, resolution, resolution);

    int cnt = 0;
    for (auto& kfp : keyframes) {
        auto kf = kfp.second;
        SE3 pose;
        if (FLAGS_pose_source == "rtk") {
            pose = kf->rtk_pose_;
        } else if (FLAGS_pose_source == "lidar") {
            pose = kf->lidar_pose_;
        } else if (FLAGS_pose_source == "opti1") {
            pose = kf->opti_pose_1_;
        } else if (FLAGS_pose_source == "opti2") {
            pose = kf->opti_pose_2_;
        }

        kf->LoadScan("./data/ch9/");

        CloudPtr cloud_trans(new PointCloudType);
        pcl::transformPointCloud(*kf->cloud_, *cloud_trans, pose.matrix());

        // voxel size
        CloudPtr kf_cloud_voxeled(new PointCloudType);
        voxel_grid_filter.setInputCloud(cloud_trans);
        voxel_grid_filter.filter(*kf_cloud_voxeled);

        *global_cloud += *kf_cloud_voxeled;
        kf->cloud_ = nullptr;

        //              << " global pts: " << global_cloud->size();
        spdlog::info("merging {} in {}, pts: {}, global pts: {}", cnt, keyframes.size(), kf_cloud_voxeled->size(),
                     global_cloud->size());
        cnt++;
    }

    if (!global_cloud->empty()) {
        sad::SaveCloudToFile(FLAGS_dump_to + "/map.pcd", *global_cloud);
    }

    spdlog::info("done.");
    return 0;
}
