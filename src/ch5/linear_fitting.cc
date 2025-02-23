#include <opencv2/opencv.hpp>
#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "common/eigen_types.h"
#include "common/math_utils.h"

DEFINE_int32(num_tested_points_plane, 10, "number of tested points in plane fitting");
DEFINE_int32(num_tested_points_line, 100, "number of tested points in line fitting");
DEFINE_double(noise_sigma, 0.01, "noise of generated samples");

void PlaneFittingTest();
void LineFittingTest();

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    spdlog::info("testing plane fitting");
    PlaneFittingTest();

    spdlog::info("testing line fitting");
    LineFittingTest();
}

void PlaneFittingTest() {
    Vec4d true_plane_coeffs(0.1, 0.2, 0.3, 0.4);
    true_plane_coeffs.normalize();

    std::vector<Vec3d> points;

    // 随机生成仿真平面点
    cv::RNG rng;
    for (int i = 0; i < FLAGS_num_tested_points_plane; ++i) {
        // 先生成一个随机点，计算第四维，增加噪声，再归一化
        Vec3d p(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
        double n4 = -p.dot(true_plane_coeffs.head<3>()) / true_plane_coeffs[3];
        p = p / (n4 + std::numeric_limits<double>::min());  // 防止除零
        p += Vec3d(rng.gaussian(FLAGS_noise_sigma), rng.gaussian(FLAGS_noise_sigma), rng.gaussian(FLAGS_noise_sigma));

        points.emplace_back(p);

        // 验证在平面上
        //
        spdlog::info("res of p: {}", p.dot(true_plane_coeffs.head<3>()) + true_plane_coeffs[3]);
    }

    Vec4d estimated_plane_coeffs;
    if (sad::math::FitPlane(points, estimated_plane_coeffs)) {
        //
        //              << ", true: " << true_plane_coeffs.transpose();
        spdlog::info("estimated coeffs: {}, true: {}", estimated_plane_coeffs.transpose(),
                     true_plane_coeffs.transpose());
    } else {
        spdlog::info("plane fitting failed");
    }
}

void LineFittingTest() {
    // 直线拟合参数真值
    Vec3d true_line_origin(0.1, 0.2, 0.3);
    Vec3d true_line_dir(0.4, 0.5, 0.6);
    true_line_dir.normalize();

    // 随机生成直线点，利用参数方程
    std::vector<Vec3d> points;
    cv::RNG rng;
    for (int i = 0; i < fLI::FLAGS_num_tested_points_line; ++i) {
        double t = rng.uniform(-1.0, 1.0);
        Vec3d p = true_line_origin + true_line_dir * t;
        p += Vec3d(rng.gaussian(FLAGS_noise_sigma), rng.gaussian(FLAGS_noise_sigma), rng.gaussian(FLAGS_noise_sigma));

        points.emplace_back(p);
    }

    Vec3d esti_origin, esti_dir;
    if (sad::math::FitLine(points, esti_origin, esti_dir)) {
        //
        //
        spdlog::info("estimated origin: {}, true: {}", esti_origin.transpose(), true_line_origin.transpose());
        spdlog::info("estimated dir: {}, true: {}", esti_dir.transpose(), true_line_dir.transpose());
    } else {
        spdlog::info("line fitting failed");
    }
}
