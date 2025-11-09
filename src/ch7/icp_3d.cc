//
// Created by xiang on 2022/7/7.
//

#include "icp_3d.h"
#include "common/math_utils.h"

#include <execution>

namespace sad {

bool Icp3d::AlignP2P(SE3& init_pose) {
    spdlog::info("aligning with point to point");
    assert(target_ != nullptr && source_ != nullptr);

    SE3 pose = init_pose;
    if (!options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
    }

    // 对点的索引，预先生成
    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    // 我们来写一些并发代码
    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Vec3d> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Vec3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 1);

            if (!nn.empty()) {
                Vec3d p = ToVec3d(target_->points[nn[0]]);
                double dis2 = (p - qs).squaredNorm();
                if (dis2 > options_.max_nn_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;

                // build residual
                Vec3d e = p - qs;
                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = pose.so3().matrix() * SO3::hat(q);
                J.block<3, 3>(0, 3) = -Mat3d::Identity();

                jacobians[idx] = J;
                errors[idx] = e;
            } else {
                effect_pts[idx] = false;
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err =
            std::accumulate(index.begin(), index.end(), std::pair<Mat6d, Vec6d>(Mat6d::Zero(), Vec6d::Zero()),
                            [&jacobians, &errors, &effect_pts, &total_res, &effective_num](
                                const std::pair<Mat6d, Vec6d>& pre, int idx) -> std::pair<Mat6d, Vec6d> {
                                if (!effect_pts[idx]) {
                                    return pre;
                                } else {
                                    auto e2 = errors[idx].dot(errors[idx]);
                                    total_res += e2;
                                    effective_num++;
                                    double w;
                                    {
                                        double delta = 1.0;
                                        double delta2 = delta * delta;
                                        double delta2_inv = 1.0 / delta2;
                                        double aux = delta2_inv * e2 + 1.0;
                                        w = 1.0 / aux;
                                    }
                                    Mat3d weighted_infos = w * Mat3d::Identity();
                                    return std::pair<Mat6d, Vec6d>(
                                        pre.first + jacobians[idx].transpose() * weighted_infos * jacobians[idx],
                                        pre.second - w * jacobians[idx].transpose() * errors[idx]);
                                }
                            });

        if (effective_num < options_.min_effective_pts_) {
            spdlog::warn("effective num too small: {}", effective_num);
            return false;
        }

        Mat6d H = H_and_err.first;
        Vec6d err = H_and_err.second;

        Vec6d dx = H.inverse() * err;
        pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        // 更新
        //              << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm();
        spdlog::info("iter {} total res: {}, eff: {}, mean res: {}, dxn: {}", iter, total_res, effective_num,
                     total_res / effective_num, dx.norm());

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            spdlog::info("iter {} pose error: {}", iter, pose_error);
        }

        if (dx.norm() < options_.eps_) {
            spdlog::info("converged, dx = {}", dx.transpose());
            break;
        }
    }

    init_pose = pose;
    return true;
}

bool Icp3d::AlignP2Plane(SE3& init_pose) {
    spdlog::info("aligning with point to plane");
    assert(target_ != nullptr && source_ != nullptr);
    // 整体流程与p2p一致，读者请关注变化部分

    SE3 pose = init_pose;
    if (!options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 1, 6>> jacobians(index.size());
    std::vector<double> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Vec3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);  // 这里取5个最近邻
            if (nn.size() > 3) {
                // convert to eigen
                std::vector<Vec3d> nn_eigen;
                for (int i = 0; i < nn.size(); ++i) {
                    nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                }

                Vec4d n;
                if (!math::FitPlane(nn_eigen, n)) {
                    // 失败的不要
                    effect_pts[idx] = false;
                    return;
                }

                double dis = n.head<3>().dot(qs) + n[3];
                if (fabs(dis) > options_.max_plane_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;

                // build residual
                Eigen::Matrix<double, 1, 6> J;
                J.block<1, 3>(0, 0) = -n.head<3>().transpose() * pose.so3().matrix() * SO3::hat(q);
                J.block<1, 3>(0, 3) = n.head<3>().transpose();

                jacobians[idx] = J;
                errors[idx] = dis;
            } else {
                effect_pts[idx] = false;
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err = std::accumulate(
            index.begin(), index.end(), std::pair<Mat6d, Vec6d>(Mat6d::Zero(), Vec6d::Zero()),
            [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Mat6d, Vec6d>& pre,
                                                                           int idx) -> std::pair<Mat6d, Vec6d> {
                if (!effect_pts[idx]) {
                    return pre;
                } else {
                    auto e2 = errors[idx] * errors[idx];
                    total_res += e2;
                    effective_num++;
                    double w;
                    {
                        double delta = 1.0;
                        double delta2 = delta * delta;
                        double delta2_inv = 1.0 / delta2;
                        double aux = delta2_inv * e2 + 1.0;
                        w = 1.0 / aux;
                    }
                    return std::pair<Mat6d, Vec6d>(pre.first + w * jacobians[idx].transpose() * jacobians[idx],
                                                   pre.second - w * jacobians[idx].transpose() * errors[idx]);
                }
            });

        if (effective_num < options_.min_effective_pts_) {
            spdlog::warn("effective num too small: {}", effective_num);
            return false;
        }

        Mat6d H = H_and_err.first;
        Vec6d err = H_and_err.second;

        Vec6d dx = H.inverse() * err;
        pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        // 更新
        //              << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm();
        spdlog::info("iter {} total res: {}, eff: {}, mean res: {}, dxn: {}", iter, total_res, effective_num,
                     total_res / effective_num, dx.norm());

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            spdlog::info("iter {} pose error: {}", iter, pose_error);
        }

        if (dx.norm() < options_.eps_) {
            spdlog::info("converged, dx = {}", dx.transpose());
            break;
        }
    }

    init_pose = pose;
    return true;
}

void Icp3d::BuildTargetKdTree() {
    kdtree_ = std::make_shared<KdTree>();
    kdtree_->BuildTree(target_);
    kdtree_->SetEnableANN();
}

bool Icp3d::AlignP2Line(SE3& init_pose) {
    spdlog::info("aligning with point to line");
    assert(target_ != nullptr && source_ != nullptr);
    // 点线与点面基本是完全一样的

    SE3 pose = init_pose;
    if (options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
        spdlog::info("init trans set to {}", pose.translation().transpose());
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Vec3d> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Vec3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);  // 这里取5个最近邻
            if (nn.size() == 5) {
                // convert to eigen
                std::vector<Vec3d> nn_eigen;
                for (int i = 0; i < 5; ++i) {
                    nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                }

                Vec3d d, p0;
                if (!math::FitLine(nn_eigen, p0, d, options_.max_line_distance_)) {
                    // 失败的不要
                    effect_pts[idx] = false;
                    return;
                }

                Vec3d err = SO3::hat(d) * (qs - p0);

                if (err.norm() > options_.max_line_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;

                // build residual
                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = -SO3::hat(d) * pose.so3().matrix() * SO3::hat(q);
                J.block<3, 3>(0, 3) = SO3::hat(d);

                jacobians[idx] = J;
                errors[idx] = err;
            } else {
                effect_pts[idx] = false;
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err =
            std::accumulate(index.begin(), index.end(), std::pair<Mat6d, Vec6d>(Mat6d::Zero(), Vec6d::Zero()),
                            [&jacobians, &errors, &effect_pts, &total_res, &effective_num](
                                const std::pair<Mat6d, Vec6d>& pre, int idx) -> std::pair<Mat6d, Vec6d> {
                                if (!effect_pts[idx]) {
                                    return pre;
                                } else {
                                    auto e2 = errors[idx].dot(errors[idx]);
                                    total_res += e2;
                                    effective_num++;
                                    double w;
                                    {
                                        double delta = 1.0;
                                        double delta2 = delta * delta;
                                        double delta2_inv = 1.0 / delta2;
                                        double aux = delta2_inv * e2 + 1.0;
                                        w = 1.0 / aux;
                                    }
                                    Mat3d weighted_infos = w * Mat3d::Identity();
                                    return std::pair<Mat6d, Vec6d>(
                                        pre.first + jacobians[idx].transpose() * weighted_infos * jacobians[idx],
                                        pre.second - w * jacobians[idx].transpose() * errors[idx]);
                                }
                            });

        if (effective_num < options_.min_effective_pts_) {
            spdlog::warn("effective num too small: {}", effective_num);
            return false;
        }

        Mat6d H = H_and_err.first;
        Vec6d err = H_and_err.second;

        Vec6d dx = H.inverse() * err;
        pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            spdlog::info("iter {} pose error: {}", iter, pose_error);
        }

        // 更新
        //              << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm();
        spdlog::info("iter {} total res: {}, eff: {}, mean res: {}, dxn: {}", iter, total_res, effective_num,
                     total_res / effective_num, dx.norm());

        if (dx.norm() < options_.eps_) {
            spdlog::info("converged, dx = {}", dx.transpose());
            break;
        }
    }

    init_pose = pose;
    return true;
}

}  // namespace sad
