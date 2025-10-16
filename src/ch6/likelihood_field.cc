//
// Created by xiang on 2022/3/18.
//

#include "ch6/g2o_types.h"
#include "ch6/likelihood_filed.h"

#include "spdlog/spdlog.h"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace sad {

LikelihoodField::LikelihoodField() {
    resolutions_ = {2.f, 8.f, 20.f};
    BuildModel();
}

void LikelihoodField::SetTargetScan(Scan2d::Ptr scan) {
    target_ = scan;
    use_occu_map_ = false;
}

void LikelihoodField::SetSourceScan(Scan2d::Ptr scan) { source_ = scan; }

void LikelihoodField::BuildModel() {
    const int range = 20;  // 生成多少个像素的模板
    for (int x = -range; x <= range; ++x) {
        for (int y = -range; y <= range; ++y) {
            model_.emplace_back(x, y, std::sqrt((x * x) + (y * y)));
        }
    }
}
void LikelihoodField::build_field() {
    assert(target_);
    // 构建似然场，不同分辨率下像素尺寸不同
    const int range_px = static_cast<int>(50.f * current_resolution_);
    field_ = cv::Mat(range_px, range_px, CV_32F, 30.0f);

    const double cx = 0.5 * field_.cols;
    const double cy = 0.5 * field_.rows;

    for (int i = 0; i < target_->ranges.size(); ++i) {
        if (target_->ranges[i] < target_->range_min || target_->ranges[i] > target_->range_max) {
            continue;
        }

        double real_angle = target_->angle_min + i * target_->angle_increment;
        double x = target_->ranges[i] * std::cos(real_angle) * current_resolution_ + cx;
        double y = target_->ranges[i] * std::sin(real_angle) * current_resolution_ + cy;

        // 在(x,y)附近填入场函数（像素距离 -> 米）
        for (auto& model_pt : model_) {
            int xx = static_cast<int>(x + model_pt.dx_);
            int yy = static_cast<int>(y + model_pt.dy_);
            if (xx >= 0 && xx < field_.cols && yy >= 0 && yy < field_.rows) {
                auto res = model_pt.residual_;
                if (field_.at<float>(yy, xx) > res) {
                    field_.at<float>(yy, xx) = res;
                }
            }
        }
    }
}

bool LikelihoodField::AlignGaussNewton(SE2& init_pose) {
    SE2 current_pose = init_pose;

    // 分辨率从粗到细迭代求解
    for (float res : resolutions_) {
        current_resolution_ = res;

        // 重建当前层的场函数
        if (use_occu_map_ && !occu_map_cache_.empty()) {
            SetFieldImageFromOccuMap(occu_map_cache_);
        } else {
            build_field();
        }

        int iterations = 10;
        double cost = 0, lastCost = 0;
        const int min_effect_pts = 20;  // 最小有效点数
        const Vec2d center(0.5 * field_.cols, 0.5 * field_.rows);
        const int image_boarder = std::max(2, static_cast<int>(std::round(border_m_ * current_resolution_)));

        has_outside_pts_ = false;
        for (int iter = 0; iter < iterations; ++iter) {
            Mat3d H = Mat3d::Zero();
            Vec3d b = Vec3d::Zero();
            cost = 0;

            int effective_num = 0;  // 有效点数（仅统计有梯度信息的点）

            // 遍历source
            for (size_t i = 0; i < source_->ranges.size(); ++i) {
                float r = source_->ranges[i];
                if (r < source_->range_min || r > source_->range_max) {
                    continue;
                }

                float angle = source_->angle_min + i * source_->angle_increment;
                if (angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
                    continue;
                }

                float theta = current_pose.so2().log();
                Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));

                // 在field中的图像坐标（以图像中心为偏移）
                Vec2i pf = (pw * current_resolution_ + center).cast<int>();

                if (pf[0] >= image_boarder && pf[0] < field_.cols - image_boarder && pf[1] >= image_boarder &&
                    pf[1] < field_.rows - image_boarder) {
                    // 图像梯度（对像素），残差单位为米
                    float dx = 0.5f * (field_.at<float>(pf[1], pf[0] + 1) - field_.at<float>(pf[1], pf[0] - 1));
                    float dy = 0.5f * (field_.at<float>(pf[1] + 1, pf[0]) - field_.at<float>(pf[1] - 1, pf[0]));

                    // 仅使用有信息的点
                    // if (dx * dx + dy * dy < grad_eps_ * grad_eps_) {
                    //     continue;
                    // }
                    effective_num++;

                    Vec3d J;
                    // 将对像素的梯度转为对米的梯度：乘以 current_resolution_
                    J << current_resolution_ * dx, current_resolution_ * dy,
                        -current_resolution_ * dx * r * std::sin(angle + theta) +
                            current_resolution_ * dy * r * std::cos(angle + theta);

                    H += J * J.transpose();

                    float e = field_.at<float>(pf[1], pf[0]);  // 米
                    b += -J * e;

                    cost += e * e;
                } else {
                    has_outside_pts_ = true;
                }
            }

            if (effective_num < min_effect_pts) {
                return false;
            }

            // solve for dx
            Vec3d dxp = H.ldlt().solve(b);
            if (isnan(dxp[0])) {
                break;
            }

            cost /= effective_num;
            if (iter > 0 && cost >= lastCost) {
                break;
            }

            current_pose.translation() += dxp.head<2>();
            current_pose.so2() = current_pose.so2() * SO2::exp(dxp[2]);
            lastCost = cost;
            cost_ = cost;
        }
        // 将当前层结果作为下一层初值
        init_pose = current_pose;
    }

    return true;
}

cv::Mat LikelihoodField::GetFieldImage() {
    cv::Mat image(field_.rows, field_.cols, CV_8UC3);
    for (int x = 0; x < field_.cols; ++x) {
        for (int y = 0; y < field_.rows; ++y) {
            float r = field_.at<float>(y, x) * 255.0 / 30.0;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(uchar(r), uchar(r), uchar(r));
        }
    }

    return image;
}

bool LikelihoodField::AlignG2O(SE2& init_pose) {
    // coarse-to-fine
    for (float res : resolutions_) {
        current_resolution_ = res;

        // 重建当前层的场函数
        if (use_occu_map_ && !occu_map_cache_.empty()) {
            SetFieldImageFromOccuMap(occu_map_cache_);
        } else {
            build_field();
        }

        using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
        using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        auto* v = new VertexSE2();
        v->setId(0);
        v->setEstimate(init_pose);
        optimizer.addVertex(v);

        const double range_th = 15.0;  // 不考虑太远的scan，不准
        const double rk_delta = 0.8;

        has_outside_pts_ = false;
        // 遍历source
        for (size_t i = 0; i < source_->ranges.size(); ++i) {
            float r = source_->ranges[i];
            if (r < source_->range_min || r > source_->range_max) {
                continue;
            }

            if (r > range_th) {
                continue;
            }

            float angle = source_->angle_min + i * source_->angle_increment;
            if (angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
                continue;
            }

            auto e = new EdgeSE2LikelihoodFiled(field_, r, angle, current_resolution_);
            e->setVertex(0, v);

            if (e->IsOutSide()) {
                has_outside_pts_ = true;
                delete e;
                continue;
            }

            e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
            auto rk = new g2o::RobustKernelHuber;
            rk->setDelta(rk_delta);
            e->setRobustKernel(rk);
            optimizer.addEdge(e);
        }

        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        init_pose = v->estimate();
        // optimizer释放由其析构处理
    }

    return true;
}

void LikelihoodField::SetFieldImageFromOccuMap(const cv::Mat& occu_map) {
    const int boarder = 25;
    use_occu_map_ = true;
    occu_map_cache_ = occu_map.clone();

    // 按当前分辨率重建 field（固定物理范围，尺寸随分辨率变化）
    const int range_px = static_cast<int>(50.f * current_resolution_);
    field_ = cv::Mat(range_px, range_px, CV_32F, 30.0f);

    const double cx = 0.5 * field_.cols;
    const double cy = 0.5 * field_.rows;
    const double ox = 0.5 * occu_map.cols;
    const double oy = 0.5 * occu_map.rows;
    const double scale = static_cast<double>(range_px) / static_cast<double>(occu_map.cols);  // 等比缩放

    for (int x = boarder; x < occu_map.cols - boarder; ++x) {
        for (int y = boarder; y < occu_map.rows - boarder; ++y) {
            if (occu_map.at<uchar>(y, x) < 127) {
                // 将 occu_map 坐标（以其中心为原点）映射到当前 field 尺寸
                const double fx = cx + (static_cast<double>(x) - ox) * scale;
                const double fy = cy + (static_cast<double>(y) - oy) * scale;

                // 在该点生成一个model（像素距离 -> 米）
                for (auto& model_pt : model_) {
                    int xx = static_cast<int>(fx + model_pt.dx_);
                    int yy = static_cast<int>(fy + model_pt.dy_);
                    if (xx >= 0 && xx < field_.cols && yy >= 0 && yy < field_.rows) {
                        float res_m = model_pt.residual_ / current_resolution_;
                        if (field_.at<float>(yy, xx) > res_m) {
                            field_.at<float>(yy, xx) = res_m;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace sad
