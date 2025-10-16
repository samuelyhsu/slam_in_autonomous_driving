//
// Created by xiang on 2022/3/18.
//

#ifndef SLAM_IN_AUTO_DRIVING_LIKELIHOOD_FILED_H
#define SLAM_IN_AUTO_DRIVING_LIKELIHOOD_FILED_H

#include <opencv2/core.hpp>
#include <vector>

#include "common/eigen_types.h"
#include "common/lidar_utils.h"

namespace sad {

class LikelihoodField {
   public:
    /// 2D 场的模板，在设置target scan或map的时候生成
    struct ModelPoint {
        ModelPoint(int dx, int dy, float res) : dx_(dx), dy_(dy), residual_(res) {}
        int dx_ = 0;
        int dy_ = 0;
        float residual_ = 0;
    };

    LikelihoodField();

    /// 增加一个2D的目标scan
    void SetTargetScan(Scan2d::Ptr scan);

    /// 设置被配准的那个scan
    void SetSourceScan(Scan2d::Ptr scan);

    /// 从占据栅格地图生成一个似然场地图
    void SetFieldImageFromOccuMap(const cv::Mat& occu_map);

    /// 使用高斯牛顿法配准
    bool AlignGaussNewton(SE2& init_pose);

    /**
     * 使用g2o配准
     * @param init_pose 初始位姿 NOTE 使用submap时，给定相对于该submap的位姿，估计结果也是针对于这个submap的位姿
     * @return
     */
    bool AlignG2O(SE2& init_pose);

    /// 获取场函数，转换为RGB图像
    cv::Mat GetFieldImage();

    bool HasOutsidePoints() const { return has_outside_pts_; }

    void SetPose(const SE2& pose) { pose_ = pose; }

    float get_cost() const { return cost_; }

   private:
    void BuildModel();
    void build_field();

    SE2 pose_;  // T_W_S
    Scan2d::Ptr target_ = nullptr;
    Scan2d::Ptr source_ = nullptr;

    std::vector<ModelPoint> model_;  // 2D 模板
    cv::Mat field_;                  // 场函数
    bool has_outside_pts_ = false;   // 是否含有出了这个场的点

    // 多层金字塔配置
    std::vector<float> resolutions_;
    float current_resolution_;  // 当前层像素/米

    // 可选：占据栅格缓存，以便多层重建
    cv::Mat occu_map_cache_;
    bool use_occu_map_ = false;

    // 以米为单位的图像边界（用于梯度的有效像素区），各层按分辨率换算为像素
    float border_m_ = 1.0f;

    // 梯度幅值阈值（像素梯度单位），低于此阈值的点视为无信息
    float grad_eps_ = 1e-3f;

    // 参数配置
    // inline static const float resolution_ = 20;  // 移除：由 current_resolution_ 取代

    float cost_;
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_LIKELIHOOD_FILED_H
