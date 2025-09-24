//
// Created by xiang on 2021/8/19.
//
#include <gtest/gtest.h>
#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

#include "ch5/bfnn.h"
#include "ch5/gridnn.hpp"
#include "ch5/kdtree.h"
#include "ch5/octo_tree.h"
#include "common/point_cloud_utils.h"
#include "common/point_types.h"
#include "common/sys_utils.h"

DEFINE_string(first_scan_path, "./data/ch5/first.pcd", "第一个点云路径");
DEFINE_string(second_scan_path, "./data/ch5/second.pcd", "第二个点云路径");
DEFINE_double(ANN_alpha, 1.0, "AAN的比例因子");

TEST(CH5_TEST, BFNN) {
    sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
    pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
    pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

    if (first->empty() || second->empty()) {
        spdlog::error("cannot load cloud");
        FAIL();
    }

    // voxel grid 至 0.05
    sad::VoxelGrid(first);
    sad::VoxelGrid(second);

    //

    // 评价单线程和多线程版本的暴力匹配
    sad::evaluate_and_call(
        [&first, &second]() {
            std::vector<std::pair<size_t, size_t>> matches;
            sad::bfnn_cloud(first, second, matches);
        },
        "暴力匹配（单线程）", 5);
    sad::evaluate_and_call(
        [&first, &second]() {
            std::vector<std::pair<size_t, size_t>> matches;
            sad::bfnn_cloud_mt(first, second, matches);
        },
        "暴力匹配（多线程）", 5);

    SUCCEED();
}

/**
 * 评测最近邻的正确性
 * @param truth 真值
 * @param esti  估计
 */
void EvaluateMatches(const std::vector<std::pair<size_t, size_t>>& truth,
                     const std::vector<std::pair<size_t, size_t>>& esti) {
    int fp = 0;  // false-positive，esti存在但truth中不存在
    int fn = 0;  // false-negative, truth存在但esti不存在

    //
    spdlog::info("truth: {}, esti: {}", truth.size(), esti.size());

    /// 检查某个匹配在另一个容器中存不存在
    auto exist = [](const std::pair<size_t, size_t>& data, const std::vector<std::pair<size_t, size_t>>& vec) -> bool {
        return std::find(vec.begin(), vec.end(), data) != vec.end();
    };

    int effective_esti = 0;
    for (const auto& d : esti) {
        if (d.first != sad::math::kINVALID_ID && d.second != sad::math::kINVALID_ID) {
            effective_esti++;

            if (!exist(d, truth)) {
                fp++;
            }
        }
    }

    for (const auto& d : truth) {
        if (!exist(d, esti)) {
            fn++;
        }
    }

    float precision = 1.0 - float(fp) / effective_esti;
    float recall = 1.0 - float(fn) / truth.size();
    //
    spdlog::info("precision: {}, recall: {}, fp: {}, fn: {}", precision, recall, fp, fn);
}

TEST(CH5_TEST, GRID_NN) {
    sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
    pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
    pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

    if (first->empty() || second->empty()) {
        spdlog::error("cannot load cloud");
        FAIL();
    }

    // voxel grid 至 0.05
    sad::VoxelGrid(first);
    sad::VoxelGrid(second);

    //
    spdlog::info("points: {}, {}", first->size(), second->size());

    std::vector<std::pair<size_t, size_t>> truth_matches;
    sad::bfnn_cloud(first, second, truth_matches);

    // 对比不同种类的grid
    sad::GridNN<2> grid2_0(0.1, sad::GridNN<2>::NearbyType::CENTER);
    sad::GridNN<2> grid2_4(0.1, sad::GridNN<2>::NearbyType::NEARBY4);
    sad::GridNN<2> grid2_8(0.1, sad::GridNN<2>::NearbyType::NEARBY8);
    sad::GridNN<3> grid3_6(0.1, sad::GridNN<3>::NearbyType::NEARBY6);
    sad::GridNN<3> grid3_14(0.1, sad::GridNN<3>::NearbyType::NEARBY14);

    grid2_0.SetPointCloud(first);
    grid2_4.SetPointCloud(first);
    grid2_8.SetPointCloud(first);
    grid3_6.SetPointCloud(first);
    grid3_14.SetPointCloud(first);

    // 评价各种版本的Grid NN

    auto evaluate = [&](auto& method, const std::string& name) {
        spdlog::info("===================");
        std::vector<std::pair<size_t, size_t>> matches;
        sad::evaluate_and_call(
            [&first, &second, &method, &matches]() { method.GetClosestPointForCloud(first, second, matches); }, name,
            10);
        EvaluateMatches(truth_matches, matches);

        sad::evaluate_and_call(
            [&first, &second, &method, &matches]() { method.GetClosestPointForCloudMT(first, second, matches); },
            name + "_mt", 10);
        EvaluateMatches(truth_matches, matches);
    };

    evaluate(grid2_0, "grid2_0");
    evaluate(grid2_4, "grid2_4");
    evaluate(grid2_8, "grid2_8");
    evaluate(grid3_6, "grid3_6");
    evaluate(grid3_14, "grid3_14");

    SUCCEED();
}

TEST(CH5_TEST, KDTREE_BASICS) {
    sad::CloudPtr cloud(new sad::PointCloudType);
    sad::PointType p1, p2, p3, p4;
    p1.x = 0;
    p1.y = 0;
    p1.z = 0;

    p2.x = 1;
    p2.y = 0;
    p2.z = 0;

    p3.x = 0;
    p3.y = 1;
    p3.z = 0;

    p4.x = 1;
    p4.y = 1;
    p4.z = 0;

    cloud->points.push_back(p1);
    cloud->points.push_back(p2);
    cloud->points.push_back(p3);
    cloud->points.push_back(p4);

    sad::KdTree kdtree;
    kdtree.BuildTree(cloud);
    kdtree.PrintAll();

    SUCCEED();
}

#include <nanoflann.hpp>

template <typename T>
struct PointCloud_NanoFlann {
    struct Point {
        T x, y, z;
    };
    using coord_t = T;  //!< The type of each coordinate
    std::vector<Point> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }
};

using kdtree_nano =
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud_NanoFlann<float>>,
                                        PointCloud_NanoFlann<float>, 3>;

void BuildNanoFlannTree(const sad::CloudPtr& cloud, std::unique_ptr<kdtree_nano>& tree,
                        PointCloud_NanoFlann<float>& cloud_flann) {
    cloud_flann.pts.resize(cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); i++) {
        cloud_flann.pts[i].x = cloud->points[i].x;
        cloud_flann.pts[i].y = cloud->points[i].y;
        cloud_flann.pts[i].z = cloud->points[i].z;
    }
    tree = std::make_unique<kdtree_nano>(3, cloud_flann, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree->buildIndex();
}

template <bool USE_FIND_NEIGHBORS>
void NanoFlannKnnSearch(const sad::CloudPtr& query_cloud, kdtree_nano* tree, int k,
                        std::vector<std::pair<size_t, size_t>>& matches) {
    matches.clear();
    std::vector<int> indices(query_cloud->size());
    for (size_t i = 0; i < query_cloud->points.size(); ++i) indices[i] = i;
    using IndexType = std::conditional_t<USE_FIND_NEIGHBORS, size_t, uint32_t>;
    std::vector<std::vector<IndexType>> ret_index_all(query_cloud->size());
    std::vector<std::vector<float>> out_dist_sqr_all(query_cloud->size());
    std::for_each(
        std::execution::par_unseq, indices.begin(), indices.end(),
        [&query_cloud, tree, k, &ret_index_all, &out_dist_sqr_all](int idx) {
            IndexType ret_index[k];
            float out_dist_sqr[k];
            float query_p[3] = {query_cloud->points[idx].x, query_cloud->points[idx].y, query_cloud->points[idx].z};
            if constexpr (USE_FIND_NEIGHBORS) {
                nanoflann::KNNResultSet<float> resultSet(k);
                resultSet.init(ret_index, out_dist_sqr);
                tree->findNeighbors(resultSet, query_p);
            } else {
                tree->knnSearch(&query_p[0], k, ret_index, out_dist_sqr);
            }
            ret_index_all[idx] = std::vector<IndexType>(ret_index, ret_index + k);
            out_dist_sqr_all[idx] = std::vector<float>(out_dist_sqr, out_dist_sqr + k);
        });
    for (size_t i = 0; i < query_cloud->points.size(); i++) {
        for (size_t j = 0; j < ret_index_all[i].size(); ++j) {
            matches.push_back({ret_index_all[i][j], i});
        }
    }
}

void test_nanoflann(sad::CloudPtr& first, sad::CloudPtr& second,
                    const std::vector<std::pair<size_t, size_t>>& true_matches) {
    using kdtree_nano =
        nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud_NanoFlann<float>>,
                                            PointCloud_NanoFlann<float>, 3>;
    spdlog::info("building kdtree nanflann");
    std::unique_ptr<kdtree_nano> my_kdtree_nano;
    PointCloud_NanoFlann<float> first_cloud_flann;
    sad::evaluate_and_call([&first, &my_kdtree_nano,
                            &first_cloud_flann]() { BuildNanoFlannTree(first, my_kdtree_nano, first_cloud_flann); },
                           "Kd Tree build nanflann", 1);
    spdlog::info("searching nanflann");
    std::vector<std::pair<size_t, size_t>> matches;
    int k = 5;
    sad::evaluate_and_call([&second, &my_kdtree_nano, &matches,
                            &k]() { NanoFlannKnnSearch<false>(second, my_kdtree_nano.get(), k, matches); },
                           "Kd Tree 5NN nanflann::knnSearch", 1);
    EvaluateMatches(true_matches, matches);
    my_kdtree_nano.reset();

    spdlog::info("building kdtree nanflann 2");
    std::unique_ptr<kdtree_nano> my_kdtree_nano2;
    sad::evaluate_and_call([&first, &my_kdtree_nano2,
                            &first_cloud_flann]() { BuildNanoFlannTree(first, my_kdtree_nano2, first_cloud_flann); },
                           "nanflann Kd Tree build nanflann", 1);
    spdlog::info("searching nanflann 2");
    sad::evaluate_and_call([&second, &my_kdtree_nano2, &matches,
                            &k]() { NanoFlannKnnSearch<true>(second, my_kdtree_nano2.get(), k, matches); },
                           "Kd Tree 5NN nanflann::findNeighbors", 1);
    EvaluateMatches(true_matches, matches);
    my_kdtree_nano2.reset();
}

TEST(CH5_TEST, KDTREE_KNN) {
    sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
    pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
    pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

    if (first->empty() || second->empty()) {
        spdlog::error("cannot load cloud");
        FAIL();
    }

    // voxel grid 至 0.05
    sad::VoxelGrid(first);
    sad::VoxelGrid(second);

    sad::KdTree kdtree;
    sad::evaluate_and_call([&first, &kdtree]() { kdtree.BuildTree(first); }, "Kd Tree build", 1);

    kdtree.SetEnableANN(true, FLAGS_ANN_alpha);

    //
    spdlog::info("Kd tree leaves: {}, points: {}", kdtree.size(), first->size());

    // 比较 bfnn
    std::vector<std::pair<size_t, size_t>> true_matches;
    sad::bfnn_cloud_mt_k(first, second, true_matches);

    // 对第2个点云执行knn
    std::vector<std::pair<size_t, size_t>> matches;
    sad::evaluate_and_call([&first, &second, &kdtree, &matches]() { kdtree.GetClosestPointMT(second, matches, 5); },
                           "Kd Tree 5NN 多线程", 1);
    EvaluateMatches(true_matches, matches);

    spdlog::info("building kdtree pcl");
    // 对比PCL
    pcl::search::KdTree<sad::PointType> kdtree_pcl;
    sad::evaluate_and_call([&first, &kdtree_pcl]() { kdtree_pcl.setInputCloud(first); }, "Kd Tree build", 1);

    spdlog::info("searching pcl");
    matches.clear();
    std::vector<int> search_indices(second->size());
    for (int i = 0; i < second->points.size(); i++) {
        search_indices[i] = i;
    }

    std::vector<std::vector<int>> result_index;
    std::vector<std::vector<float>> result_distance;
    sad::evaluate_and_call(
        [&]() { kdtree_pcl.nearestKSearch(*second, search_indices, 5, result_index, result_distance); },
        "Kd Tree 5NN in PCL", 1);
    for (int i = 0; i < second->points.size(); i++) {
        for (int j = 0; j < result_index[i].size(); ++j) {
            int m = result_index[i][j];
            double d = result_distance[i][j];
            matches.push_back({m, i});
        }
    }
    EvaluateMatches(true_matches, matches);

    test_nanoflann(first, second, true_matches);
    spdlog::info("done.");

    SUCCEED();
}

TEST(CH5_TEST, OCTREE_BASICS) {
    sad::CloudPtr cloud(new sad::PointCloudType);
    sad::PointType p1, p2, p3, p4;
    p1.x = 0;
    p1.y = 0;
    p1.z = 0;

    p2.x = 1;
    p2.y = 0;
    p2.z = 0;

    p3.x = 0;
    p3.y = 1;
    p3.z = 0;

    p4.x = 1;
    p4.y = 1;
    p4.z = 0;

    cloud->points.push_back(p1);
    cloud->points.push_back(p2);
    cloud->points.push_back(p3);
    cloud->points.push_back(p4);

    sad::OctoTree octree;
    octree.BuildTree(cloud);
    octree.SetApproximate(false);
    //
    spdlog::info("Octo tree leaves: {}, points: {}", octree.size(), cloud->size());

    SUCCEED();
}

TEST(CH5_TEST, OCTREE_KNN) {
    sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
    pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
    pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

    if (first->empty() || second->empty()) {
        spdlog::error("cannot load cloud");
        FAIL();
    }

    // voxel grid 至 0.05
    sad::VoxelGrid(first);
    sad::VoxelGrid(second);

    sad::OctoTree octree;
    sad::evaluate_and_call([&first, &octree]() { octree.BuildTree(first); }, "Octo Tree build", 1);

    octree.SetApproximate(true, FLAGS_ANN_alpha);
    //
    spdlog::info("Octo tree leaves: {}, points: {}", octree.size(), first->size());

    /// 测试KNN
    spdlog::info("testing knn");
    std::vector<std::pair<size_t, size_t>> matches;
    sad::evaluate_and_call([&first, &second, &octree, &matches]() { octree.GetClosestPointMT(second, matches, 5); },
                           "Octo Tree 5NN 多线程", 1);

    spdlog::info("comparing with bfnn");
    /// 比较真值
    std::vector<std::pair<size_t, size_t>> true_matches;
    sad::bfnn_cloud_mt_k(first, second, true_matches);
    EvaluateMatches(true_matches, matches);

    spdlog::info("done.");

    SUCCEED();
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
