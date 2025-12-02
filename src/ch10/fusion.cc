//
// Created by xiang on 22-12-20.
//
#include <yaml-cpp/yaml.h>
#include <execution>

#include "common/lidar_utils.h"
#include "fusion.h"

namespace sad {

Fusion::Fusion(const std::string& config_yaml) {
    config_yaml_ = config_yaml;
    StaticIMUInit::Options imu_init_options;
    imu_init_options.use_speed_for_static_checking_ = false;  // 本节数据不需要轮速计
    imu_init_ = StaticIMUInit(imu_init_options);
    ndt_pcl_.setResolution(1.0);
    ndt_.SetResolution(1.0);
}

bool Fusion::Init() {
    // 地图原点
    auto yaml = YAML::LoadFile(config_yaml_);
    auto origin_data = yaml["origin"].as<std::vector<double>>();
    map_origin_ = Vec3d(origin_data[0], origin_data[1], origin_data[2]);

    rtk_search_min_score_ = yaml["loop_closing"]["ndt_score_th"].as<double>();
    use_pcl_ndt_ = yaml["use_pcl_ndt"].as<bool>();
    data_dir_ = yaml["data_dir"].as<std::string>();

    // 地图目录
    data_path_ = yaml["map_data"].as<std::string>();

    LoadMapIndex();

    // lidar和IMU消息同步
    sync_ = std::make_shared<MessageSync>([this](const MeasureGroup& m) { ProcessMeasurements(m); });
    sync_->Init(config_yaml_);

    // lidar和IMU外参
    std::vector<double> ext_t = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
    std::vector<double> ext_r = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();
    Vec3d lidar_T_wrt_IMU = math::VecFromArray(ext_t);
    Mat3d lidar_R_wrt_IMU = math::MatFromArray(ext_r);
    TIL_ = SE3(lidar_R_wrt_IMU, lidar_T_wrt_IMU);

    // ui
    ui_ = std::make_shared<ui::PangolinWindow>();
    ui_->Init();
    ui_->SetCurrentScanSize(50);
    return true;
}

void Fusion::ProcessRTK(GNSSPtr gnss) {
    gnss->utm_pose_.translation() -= map_origin_;  // 减掉地图原点
    last_gnss_ = gnss;
}

void Fusion::ProcessMeasurements(const MeasureGroup& meas) {
    measures_ = meas;

    if (imu_need_init_) {
        TryInitIMU();
        return;
    }

    /// 以下三步与LIO一致，只是align完成地图匹配工作
    if (status_ == Status::WORKING) {
        Predict();
        Undistort();
    } else {
        scan_undistort_ = measures_.lidar_;
    }

    Align();
}

void Fusion::TryInitIMU() {
    for (auto imu : measures_.imu_) {
        imu_init_.AddIMU(*imu);
    }

    if (imu_init_.InitSuccess()) {
        // 读取初始零偏，设置ESKF
        sad::ESKFD::Options options;
        // 噪声由初始化器估计
        // options.gyro_var_ = sqrt(imu_init_.GetCovGyro()[0]);
        // options.acce_var_ = sqrt(imu_init_.GetCovAcce()[0]);
        options.update_bias_acce_ = false;
        options.update_bias_gyro_ = false;
        eskf_.SetInitialConditions(options, imu_init_.GetInitBg(), imu_init_.GetInitBa(), imu_init_.GetGravity());
        imu_need_init_ = false;

        spdlog::info("IMU初始化成功");
    }
}

void Fusion::Predict() {
    imu_states_.clear();
    imu_states_.emplace_back(eskf_.GetNominalState());

    /// 对IMU状态进行预测
    for (auto& imu : measures_.imu_) {
        eskf_.Predict(*imu);
        imu_states_.emplace_back(eskf_.GetNominalState());
    }
}

void Fusion::Undistort() {
    auto cloud = measures_.lidar_;
    auto imu_state = eskf_.GetNominalState();  // 最后时刻的状态
    SE3 T_end = SE3(imu_state.R_, imu_state.p_);

    /// 将所有点转到最后时刻状态上
    std::for_each(std::execution::par_unseq, cloud->points.begin(), cloud->points.end(), [&](auto& pt) {
        SE3 Ti = T_end;
        NavStated match;

        // 根据pt.time查找时间，pt.time是该点打到的时间与雷达开始时间之差，单位为毫秒
        math::PoseInterp<NavStated>(
            measures_.lidar_begin_time_ + pt.time * 1e-3, imu_states_, [](const NavStated& s) { return s.timestamp_; },
            [](const NavStated& s) { return s.GetSE3(); }, Ti, match);

        Vec3d pi = ToVec3d(pt);
        Vec3d p_compensate = TIL_.inverse() * T_end.inverse() * Ti * TIL_ * pi;

        pt.x = p_compensate(0);
        pt.y = p_compensate(1);
        pt.z = p_compensate(2);
    });
    scan_undistort_ = cloud;
}

void Fusion::Align() {
    FullCloudPtr scan_undistort_trans(new FullPointCloudType);
    pcl::transformPointCloud(*scan_undistort_, *scan_undistort_trans, TIL_.matrix());
    scan_undistort_ = scan_undistort_trans;
    current_scan_ = ConvertToCloud<FullPointType>(scan_undistort_);
    current_scan_ = VoxelCloud(current_scan_, 0.5);

    if (status_ == Status::WAITING_FOR_RTK) {
        // 若存在最近的RTK信号，则尝试初始化
        if (last_gnss_ != nullptr) {
            if (SearchRTK()) {
                status_ == Status::WORKING;
                ui_->UpdateScan(current_scan_, eskf_.GetNominalSE3());
                ui_->UpdateNavState(eskf_.GetNominalState());
            }
        }
    } else {
        LidarLocalization();
        ui_->UpdateScan(current_scan_, eskf_.GetNominalSE3());
        ui_->UpdateNavState(eskf_.GetNominalState());
    }
}

bool Fusion::SearchRTK() {
    if (init_has_failed_) {
        if ((last_gnss_->utm_pose_.translation() - last_searched_pos_.translation()).norm() < 20.0) {
            spdlog::info("skip this position");
            return false;
        }
    }

    // 由于RTK不带姿态，我们必须先搜索一定的角度范围
    std::vector<GridSearchResult> search_poses;
    spdlog::info("{}: load map at {}", __func__, last_gnss_->utm_pose_.translation().transpose());
    LoadMap(last_gnss_->utm_pose_);

    /// 由于RTK不带角度，这里按固定步长扫描RTK角度
    double grid_ang_range = 360.0, grid_ang_step = 10;  // 角度搜索范围与步长
    for (double ang = 0; ang < grid_ang_range; ang += grid_ang_step) {
        SE3 pose(SO3::rotZ(ang * math::kDEG2RAD), Vec3d(0, 0, 0) + last_gnss_->utm_pose_.translation());
        GridSearchResult gr;
        gr.pose_ = pose;
        search_poses.emplace_back(gr);
    }

    spdlog::info("grid search poses: {}", search_poses.size());
    std::for_each(std::execution::par_unseq, search_poses.begin(), search_poses.end(),
                  [this](GridSearchResult& gr) { AlignForGrid(gr); });

    // 选择最优的匹配结果
    auto max_ele = std::max_element(search_poses.begin(), search_poses.end(),
                                    [](const auto& g1, const auto& g2) { return g1.score_ < g2.score_; });
    spdlog::info("max score: {}, pose: \n{}", max_ele->score_, max_ele->result_pose_.matrix());
    if (max_ele->score_ > rtk_search_min_score_) {
        spdlog::info("初始化成功, score: {} > {}", max_ele->score_, rtk_search_min_score_);
        status_ = Status::WORKING;

        /// 重置滤波器状态
        auto state = eskf_.GetNominalState();
        state.R_ = max_ele->result_pose_.so3();
        state.p_ = max_ele->result_pose_.translation();
        state.v_.setZero();
        eskf_.SetX(state, eskf_.GetGravity());

        ESKFD::Mat18T cov;
        cov = ESKFD::Mat18T::Identity() * 1e-4;
        cov.block<12, 12>(6, 6) = Eigen::Matrix<double, 12, 12>::Identity() * 1e-6;
        eskf_.SetCov(cov);

        return true;
    }

    init_has_failed_ = true;
    last_searched_pos_ = last_gnss_->utm_pose_;
    return false;
}

void Fusion::AlignForGrid(sad::Fusion::GridSearchResult& gr) {
    if (use_pcl_ndt_) {
        /// 多分辨率
        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(0.05);
        ndt.setStepSize(0.7);
        ndt.setMaximumIterations(40);

        ndt.setInputSource(current_scan_);
        auto map = ref_cloud_;

        CloudPtr output(new PointCloudType);
        Mat4f T = gr.pose_.matrix().cast<float>();
        for (auto& r : ndt_resolutions_) {
            auto rough_map = VoxelCloud(map, r * 0.1);
            ndt.setInputTarget(rough_map);
            ndt.setResolution(r);
            ndt.align(*output, T);
            T = ndt.getFinalTransformation();
        }
        gr.score_ = ndt.getTransformationProbability();
        gr.result_pose_ = Mat4ToSE3(T);
    } else {
        Ndt3d::Options options;
        Ndt3d ndt(options);
        ndt.SetSource(current_scan_);
        SE3 pose = gr.pose_;
        for (int i = 0; i < ndt_resolutions_.size(); ++i) {
            auto r = ndt_resolutions_[i];
            // ndt.SetTarget(ndt_resolutions_[i], ndt_surround_merged_map_[i]);
            auto rough_map = VoxelCloud(ref_cloud_, r * 0.1);
            ndt.SetResolution(r);
            ndt.SetTarget(rough_map);
            ndt.AlignNdt(pose);
        }
        gr.score_ = ndt.GetScore();
        gr.result_pose_ = pose;
    }
}

bool Fusion::LidarLocalization() {
    SE3 pose = eskf_.GetNominalSE3();
    LoadMap(pose);
    double score;
    if (use_pcl_ndt_) {
        ndt_pcl_.setInputCloud(current_scan_);
        CloudPtr output(new PointCloudType);
        ndt_pcl_.align(*output, pose.matrix().cast<float>());
        pose = Mat4ToSE3(ndt_pcl_.getFinalTransformation());
        score = ndt_pcl_.getTransformationProbability();
    } else {
        ndt_.SetSource(current_scan_);
        ndt_.AlignNdt(pose);
        score = ndt_.GetScore();
    }
    eskf_.ObserveSE3(pose, 1e-1, 1e-2);
    spdlog::info("lidar loc score: {}", score);
    return true;
}

void Fusion::LoadMap(const SE3& pose) {
    int gx = floor((pose.translation().x() - 50.0) / 100);
    int gy = floor((pose.translation().y() - 50.0) / 100);
    Vec2i key(gx, gy);

    // 一个区域的周边地图，我们认为9个就够了
    std::set<Vec2i, less_vec<2>> surrounding_index{
        key + Vec2i(0, 0), key + Vec2i(-1, 0), key + Vec2i(-1, -1), key + Vec2i(-1, 1), key + Vec2i(0, -1),
        key + Vec2i(0, 1), key + Vec2i(1, 0),  key + Vec2i(1, -1),  key + Vec2i(1, 1),
    };

    // 加载必要区域
    bool map_data_changed = false;
    int cnt_new_loaded = 0, cnt_unload = 0;
    for (auto& k : surrounding_index) {
        if (map_data_index_.find(k) == map_data_index_.end()) {
            // 该地图数据不存在
            continue;
        }

        if (map_data_.find(k) == map_data_.end()) {
            // 加载这个区块
            CloudPtr cloud(new PointCloudType);
            pcl::io::loadPCDFile(data_path_ + std::to_string(k[0]) + "_" + std::to_string(k[1]) + ".pcd", *cloud);
            map_data_.emplace(k, cloud);
            map_data_changed = true;
            cnt_new_loaded++;
        }
    }

    // 卸载不需要的区域，这个稍微加大一点，不需要频繁卸载
    for (auto iter = map_data_.begin(); iter != map_data_.end();) {
        if ((iter->first - key).cast<float>().norm() > 3.0) {
            // 卸载本区块
            iter = map_data_.erase(iter);
            cnt_unload++;
            map_data_changed = true;
        } else {
            iter++;
        }
    }

    // spdlog::info("new loaded: {}, unload: {}", cnt_new_loaded, cnt_unload);
    if (map_data_changed) {
        ref_cloud_.reset(new PointCloudType);
        for (auto& mp : map_data_) {
            *ref_cloud_ += *mp.second;
        }
        spdlog::info("rebuild global cloud, grids: {}", map_data_.size());

        if (use_pcl_ndt_) {
            ndt_pcl_.setInputTarget(ref_cloud_);
        } else {
            ndt_.SetTarget(ref_cloud_);
        }
    }

    // ui绘制当前参与定位计算的局部子地图
    ui_->UpdatePointCloudGlobal(map_data_);

#if 0
    if (!use_pcl_ndt_) {
        // 完成多分辨率自定义ndt地图的加载
        for (int i = 0; i < ndt_resolutions_.size(); ++i) {
            // 分辨率
            auto r = ndt_resolutions_[i];
            // 子地图索引
            auto& sub_map_indexes = ndt_sub_map_indexes_[i];
            // 当前加载的周边子地图
            auto& surround_sub_maps = ndt_surround_sub_maps_[i];
            // 当前加载的周边子地图合并后的地图，用于配准
            auto& surround_merged_map = ndt_surround_merged_map_[i];

            bool map_data_changed = false;
            int cnt_new_loaded = 0, cnt_unload = 0;
            // 加载周边子地图
            for (auto& sub_map_index : surrounding_index) {
                if (sub_map_indexes.find(sub_map_index) == sub_map_indexes.end()) {
                    // 不存在此子地图
                    continue;
                }
                if (surround_sub_maps.find(sub_map_index) != surround_sub_maps.end()) {
                    // 子地图已经加载
                    continue;
                }
                auto file_name =
                    fmt::format("{}/ndt_map_data_{}/{}_{}.txt", data_dir_, r, sub_map_index[0], sub_map_index[1]);
                std::ifstream fin(file_name);
                if (!fin) {
                    spdlog::critical("failed to open file: {}", file_name);
                    exit(-1);
                }
                NdtMap sub_map;
                int x, y, z, count;
                double mu_x, mu_y, mu_z;
                double info_00, info_01, info_02, info_10, info_11, info_12, info_20, info_21, info_22;
                Eigen::Matrix<int, 3, 1> key_ndt;
                Vec3d mu;
                Mat3d info;
                while (!fin.eof()) {
                    fin >> x >> y >> z >> count >> mu_x >> mu_y >> mu_z >> info_00 >> info_01 >> info_02 >> info_10 >>
                        info_11 >> info_12 >> info_20 >> info_21 >> info_22;
                    key_ndt << x, y, z;
                    mu = Vec3d(mu_x, mu_y, mu_z);
                    info << info_00, info_01, info_02, info_10, info_11, info_12, info_20, info_21, info_22;
                    if (sub_map.find(key_ndt) == sub_map.end()) {
                        Ndt3d::VoxelData voxelData;
                        voxelData.mu_ = mu;
                        voxelData.info_ = info;
                        sub_map.insert(std::make_pair(key_ndt, voxelData));  // 若不存在，则插入该栅格
                    }
                }
                surround_sub_maps.emplace(sub_map_index, sub_map);
                map_data_changed = true;
                cnt_new_loaded++;
            }

            // 卸载不需要的子地图
            for (auto it = surround_sub_maps.begin(); it != surround_sub_maps.end();) {
                if ((it->first - key).cast<float>().norm() > 3.0f) {
                    it = surround_sub_maps.erase(it);
                    cnt_unload++;
                    map_data_changed = true;
                } else {
                    it++;
                }
            }

            if (map_data_changed) {
                // spdlog::info("new loaded: {}, unload: {}", cnt_new_loaded, cnt_unload);
                surround_merged_map.clear();
                for (auto& sub_map : surround_sub_maps) {
                    auto& d = sub_map.second;
                    surround_merged_map.insert(d.begin(), d.end());
                }

                if (r == 1) {
                    // 用于后面的配准
                    ndt_.SetTarget(r, surround_merged_map);
                }
            }
        }
    }
#endif
}

void Fusion::LoadMapIndex() {
    {
        std::ifstream fin(data_dir_ + "/map_data/map_index.txt");
        while (!fin.eof()) {
            int x, y;
            fin >> x >> y;
            map_data_index_.emplace(Vec2i(x, y));
        }
        fin.close();
    }
    for (int i = 0; i < ndt_resolutions_.size(); i++) {
        auto r = ndt_resolutions_[i];
        std::ifstream fin(fmt::format("{}/ndt_map_data_{}/map_index.txt", data_dir_, r));
        while (!fin.eof()) {
            int x, y;
            fin >> x >> y;
            ndt_sub_map_indexes_[i].emplace(Vec2i(x, y));
        }
    }
}

void Fusion::ProcessIMU(IMUPtr imu) { sync_->ProcessIMU(imu); }

void Fusion::ProcessPointCloud(sensor_msgs::PointCloud2::Ptr cloud) { sync_->ProcessCloud(cloud); }

}  // namespace sad
