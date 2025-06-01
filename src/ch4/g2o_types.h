//
// Created by xiang on 23-1-19.
//

#ifndef SLAM_IN_AUTO_DRIVING_CH4_G2O_TYPES_H
#define SLAM_IN_AUTO_DRIVING_CH4_G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/robust_kernel.h>
#include "ch4/imu_preintegration.h"
#include "common/eigen_types.h"
#include "common/g2o_types.h"

namespace sad {

/// 与预积分相关的vertex, edge
/**
 * 预积分边
 * 连接6个顶点：上一帧的pose, v, bg, ba，下一帧的pose, v
 * 观测量为9维，即预积分残差, 顺序：R, v, p
 * information从预积分类中获取，构造函数中计算
 */
class EdgeInertial : public g2o::BaseMultiEdge<9, Vec9d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * 构造函数中需要指定预积分类对象
     * @param preinteg  预积分对象指针
     * @param gravity   重力矢量
     * @param weight    权重
     */
    EdgeInertial(std::shared_ptr<IMUPreintegration> preinteg, const Vec3d& gravity, double weight = 1.0);

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    void computeError() override;
    void linearizeOplus() override;

    Eigen::Matrix<double, 24, 24> GetHessian() {
        linearizeOplus();
        Eigen::Matrix<double, 9, 24> J;
        J.block<9, 6>(0, 0) = _jacobianOplus[0];
        J.block<9, 3>(0, 6) = _jacobianOplus[1];
        J.block<9, 3>(0, 9) = _jacobianOplus[2];
        J.block<9, 3>(0, 12) = _jacobianOplus[3];
        J.block<9, 6>(0, 15) = _jacobianOplus[4];
        J.block<9, 3>(0, 21) = _jacobianOplus[5];
        return J.transpose() * information() * J;
    }

   private:
    const double dt_;
    std::shared_ptr<IMUPreintegration> preint_ = nullptr;
    Vec3d grav_;
};

// v_wheel = R^T * v
// d_v_wheel/d_R = SO3::hat(R^T * v)
// d_v_wheel/d_v = R^T
class EdgeWheelSpeed : public g2o::BaseBinaryEdge<3, Vec3d, VertexVelocity, VertexPose> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeWheelSpeed() = default;

    EdgeWheelSpeed(VertexVelocity* v0, VertexPose* v1, const Vec3d& v_wheel) {
        setVertex(0, v0);
        setVertex(1, v1);
        setMeasurement(v_wheel);
    }

    void computeError() override {
        auto v0 = (VertexVelocity*)_vertices[0];
        auto v1 = (VertexPose*)_vertices[1];
        const auto& R = v1->estimate().so3();
        const auto& v = v0->estimate();
        _error = R.inverse() * v - _measurement;
    }

    void linearizeOplus() override {
        auto v0 = (VertexVelocity*)_vertices[0];
        auto v1 = (VertexPose*)_vertices[1];
        const auto& R = v1->estimate().so3();
        const auto& v = v0->estimate();
        _jacobianOplusXi = R.inverse().matrix();
        _jacobianOplusXj.setZero();
        _jacobianOplusXj.block<3, 3>(0, 0) = SO3::hat(R.inverse() * v);
    }

    // pose + velocity
    Eigen::Matrix<double, 9, 9> GetHessian() {
        linearizeOplus();
        Eigen::Matrix<double, 3, 9> J;
        J.block<3, 6>(0, 0) = _jacobianOplusXj;
        J.block<3, 3>(0, 6) = _jacobianOplusXi;
        return J.transpose() * information() * J;
    }

    virtual bool read(std::istream& in) { return true; }
    virtual bool write(std::ostream& out) const { return true; }
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
