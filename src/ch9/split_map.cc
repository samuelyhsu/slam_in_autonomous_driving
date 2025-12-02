//
// Created by xiang on 22-12-20.
//

#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <execution>
#include "ch7/ndt_3d.h"
#include "common/eigen_types.h"
#include "common/math_utils.h"
#include "common/point_cloud_utils.h"
#include "keyframe.h"

DEFINE_string(map_path, "./data/ch9/", "导出数据的目录");
DEFINE_double(voxel_size, 0.1, "导出地图分辨率");

namespace sad {

using grids_t = std::unordered_map<Ndt3d::KeyType, Ndt3d::VoxelData, hash_vec<3>>;

void BuildNdtMapVoxels(CloudPtr target_, grids_t& grids_, double voxel_size) {
    assert(target_ != nullptr);         // 目标点云指针不能为空
    assert(target_->empty() == false);  // 目标点云不能为空
    grids_.clear();                     // 清空体素栅格

    double inv_voxel_size = 1.0 / voxel_size;

    /// 分配体素索引
    std::vector<size_t> index(target_->size());
    std::for_each(index.begin(), index.end(), [idx = 0](size_t& i) mutable { i = idx++; });

    // 生成体素栅格
    std::for_each(index.begin(), index.end(), [&](const size_t& idx) {
        auto pt = ToVec3d(target_->points[idx]);
        // 对目标点云中的每个点，计算它所在的体素栅格ID
        auto key = (pt * inv_voxel_size).cast<int>();

        // 查看该栅格是否已存在
        if (grids_.find(key) == grids_.end()) grids_.insert({key, {idx}});  // 若不存在，则插入该栅格
        else
            grids_[key].idx_.emplace_back(idx);  // 若存在，则将该点的索引插入到该体素栅格中
    });

    // 并发遍历所有体素栅格
    std::for_each(std::execution::par_unseq, grids_.begin(), grids_.end(), [&](auto& v) {
        // 判断体素中的点数是否大于阈值3个
        if (v.second.idx_.size() > 3) {
            // 要求至少有３个点，才会计算每个体素中的均值和协方差
            sad::math::ComputeMeanAndCov(v.second.idx_, v.second.mu_, v.second.sigma_,
                                         [&](const size_t& idx) { return ToVec3d(target_->points[idx]); });
            // SVD 检查最大与最小奇异值，限制最小奇异值
            Eigen::JacobiSVD svd(v.second.sigma_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vec3d lambda = svd.singularValues();
            if (lambda[1] < lambda[0] * 1e-3) lambda[1] = lambda[0] * 1e-3;

            if (lambda[2] < lambda[0] * 1e-3) lambda[2] = lambda[0] * 1e-3;

            Mat3d inv_lambda = Vec3d(1.0 / lambda[0], 1.0 / lambda[1], 1.0 / lambda[2]).asDiagonal();
            v.second.info_ = svd.matrixV() * inv_lambda * svd.matrixU().transpose();  // 信息矩阵
        }
    });

    // 遍历所有体素栅格
    for (auto iter = grids_.begin(); iter != grids_.end();) {
        if (iter->second.idx_.size() > 3) iter++;
        else
            iter = grids_.erase(iter);  // 删除点数不够3个的栅格
    }
}

void SaveNDTVoxelToFile(const std::string& filePath, const grids_t& grids) {
    // std::ofstream fout_ndt_map(filePath, std::ios::binary);
    std::ofstream fout_ndt_map(filePath);
    if (fout_ndt_map.is_open()) {
        // 遍历所有体素栅格
        for (const auto& voxel : grids) {
            // 将矩阵的数据拷贝到行向量中
            Eigen::VectorXd rowVector(9);
            Mat3d mat = voxel.second.info_;
            Eigen::Map<Eigen::VectorXd>(rowVector.data(), rowVector.size()) =
                Eigen::Map<Eigen::VectorXd>(mat.data(), mat.size());
            // 将体素索引、均值和协方差矩阵写入文件的一行
            fout_ndt_map << voxel.first.transpose() << " " << voxel.second.idx_.size() << " "
                         << voxel.second.mu_.transpose() << " " << rowVector.transpose() << std::endl;
        }
    } else {
        spdlog::error("Failed to open the file.");
    }
}

}  // namespace sad

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    using namespace sad;

    std::map<IdType, KFPtr> keyframes;
    if (!LoadKeyFrames("./data/ch9/keyframes.txt", keyframes)) {
        spdlog::error("failed to load keyframes");
        return 0;
    }

    std::map<Vec2i, CloudPtr, less_vec<2>> map_data;  // 以网格ID为索引的地图数据
    pcl::VoxelGrid<PointType> voxel_grid_filter;
    float resolution = FLAGS_voxel_size;
    voxel_grid_filter.setLeafSize(resolution, resolution, resolution);

    // 逻辑和dump map差不多，但每个点个查找它的网格ID，没有的话会创建
    for (auto& kfp : keyframes) {
        auto kf = kfp.second;
        kf->LoadScan("./data/ch9/");

        CloudPtr cloud_trans(new PointCloudType);
        pcl::transformPointCloud(*kf->cloud_, *cloud_trans, kf->opti_pose_2_.matrix());

        // voxel size
        CloudPtr kf_cloud_voxeled(new PointCloudType);
        voxel_grid_filter.setInputCloud(cloud_trans);
        voxel_grid_filter.filter(*kf_cloud_voxeled);

        spdlog::info("building kf {} in {}", kf->id_, keyframes.size());

        // add to grid
        for (const auto& pt : kf_cloud_voxeled->points) {
            int gx = floor((pt.x - 50.0) / 100);
            int gy = floor((pt.y - 50.0) / 100);
            Vec2i key(gx, gy);
            auto iter = map_data.find(key);
            if (iter == map_data.end()) {
                // create point cloud
                CloudPtr cloud(new PointCloudType);
                cloud->points.emplace_back(pt);
                cloud->is_dense = false;
                cloud->height = 1;
                map_data.emplace(key, cloud);
            } else {
                iter->second->points.emplace_back(pt);
            }
        }
    }

    // 存储点云和索引文件
    spdlog::info("saving maps, grids: {}", map_data.size());
    std::system("mkdir -p ./data/ch9/map_data/");
    std::system("rm -rf ./data/ch9/map_data/*");  // 清理一下文件夹
    std::ofstream fout("./data/ch9/map_data/map_index.txt");

    constexpr auto NDT_RESOLUTION_COUNT = 4;
    std::vector<float> ndt_resolutions{10, 5, 2, 1};
    std::vector<std::ofstream> ndt_fouts;
    std::vector<grids_t> ndt_grids(NDT_RESOLUTION_COUNT);
    for (auto r : ndt_resolutions) {
        std::system(fmt::format("mkdir -p ./data/ch9/ndt_map_data_{}/", r).c_str());
        std::system(fmt::format("rm -rf ./data/ch9/ndt_map_data_{}/*", r).c_str());

        ndt_fouts.emplace_back(std::ofstream(fmt::format("./data/ch9/ndt_map_data_{}/map_index.txt", r)));
    }

    for (auto& dp : map_data) {
        fout << dp.first[0] << " " << dp.first[1] << std::endl;
        dp.second->width = dp.second->size();
        sad::VoxelGrid(dp.second, 0.1);

        sad::SaveCloudToFile(
            "./data/ch9/map_data/" + std::to_string(dp.first[0]) + "_" + std::to_string(dp.first[1]) + ".pcd",
            *dp.second);

        for (int i = 0; i < NDT_RESOLUTION_COUNT; ++i) {
            auto resolution = ndt_resolutions[i];
            auto& ndt_fout = ndt_fouts[i];
            auto& grids = ndt_grids[i];

            grids.clear();

            BuildNdtMapVoxels(dp.second, grids, resolution);

            if (!grids.empty()) {
                ndt_fout << dp.first[0] << " " << dp.first[1] << std::endl;
                auto file_name =
                    fmt::format("./data/ch9/ndt_map_data_{}/{}_{}.txt", resolution, dp.first[0], dp.first[1]);
                SaveNDTVoxelToFile(file_name, grids);
            }
        }
    }
    fout.close();

    for (auto& ndt_fout : ndt_fouts) {
        ndt_fout.close();
    }

    spdlog::info("done.");
    return 0;
}
