//
// Created by xiang on 2021/11/11.
//

#include "ch3/static_imu_init.h"
#include "common/math_utils.h"

#include "spdlog/spdlog.h"

namespace sad {

bool StaticIMUInit::AddIMU(const IMU& imu) {
    if (init_success_) {
        return true;
    }

    if (options_.use_speed_for_static_checking_ && !is_static_) {
        spdlog::warn("等待车辆静止");
        init_imu_deque_.clear();
        return false;
    }

    if (init_imu_deque_.empty()) {
        // 记录初始静止时间
        init_start_time_ = imu.timestamp_;
    }

    // 记入初始化队列
    init_imu_deque_.push_back(imu);

    double init_time = imu.timestamp_ - init_start_time_;  // 初始化经过时间
    if (init_time > options_.init_time_seconds_) {
        // 尝试初始化逻辑
        TryInit();
    }

    // 维持初始化队列长度
    while (init_imu_deque_.size() > options_.init_imu_queue_max_size_) {
        init_imu_deque_.pop_front();
    }

    current_time_ = imu.timestamp_;
    return false;
}

bool StaticIMUInit::AddOdom(const Odom& odom) {
    // 判断车辆是否静止
    if (init_success_) {
        return true;
    }

    if (odom.left_pulse_ < options_.static_odom_pulse_ && odom.right_pulse_ < options_.static_odom_pulse_) {
        is_static_ = true;
    } else {
        is_static_ = false;
    }

    current_time_ = odom.timestamp_;
    return true;
}

bool StaticIMUInit::TryInit() {
    if (init_imu_deque_.size() < 10) {
        return false;
    }

    // 计算均值和方差
    Vec3d mean_gyro, mean_acce;
    math::ComputeMeanAndCovDiag(init_imu_deque_, mean_gyro, cov_gyro_, [](const IMU& imu) { return imu.gyro_; });
    math::ComputeMeanAndCovDiag(init_imu_deque_, mean_acce, cov_acce_, [this](const IMU& imu) { return imu.acce_; });

    // 以acce均值为方向，取9.8长度为重力
    spdlog::info("mean acce: {}", mean_acce.transpose());
    gravity_ = -mean_acce / mean_acce.norm() * options_.gravity_norm_;

    // 重新计算加计的协方差
    math::ComputeMeanAndCovDiag(init_imu_deque_, mean_acce, cov_acce_,
                                [this](const IMU& imu) { return imu.acce_ + gravity_; });

    // 检查IMU噪声
    if (cov_gyro_.norm() > options_.max_static_gyro_var) {
        spdlog::error("陀螺仪测量噪声太大 {} > {}", cov_gyro_.norm(), options_.max_static_gyro_var);
        return false;
    }

    if (cov_acce_.norm() > options_.max_static_acce_var) {
        spdlog::error("加计测量噪声太大 {} > {}", cov_acce_.norm(), options_.max_static_acce_var);
        return false;
    }

    // 估计测量噪声和零偏
    init_bg_ = mean_gyro;
    init_ba_ = mean_acce;

    spdlog::info(
        "IMU 初始化成功，初始化时间= {} , bg = {} , ba = {} , gyro sq = {} , acce sq = {} , grav = {} , norm: {}",
        current_time_ - init_start_time_, init_bg_.transpose(), init_ba_.transpose(), cov_gyro_.transpose(),
        cov_acce_.transpose(), gravity_.transpose(), gravity_.norm());
    spdlog::info("mean gyro: {} acce: {}", mean_gyro.transpose(), mean_acce.transpose());
    init_success_ = true;
    return true;
}

}  // namespace sad
