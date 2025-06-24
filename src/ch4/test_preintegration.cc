//
// Created by xiang on 2021/7/16.
//

#include <gtest/gtest.h>
#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "ch3/eskf.hpp"
#include "ch3/static_imu_init.h"
#include "ch4/g2o_types.h"
#include "ch4/imu_preintegration.h"
#include "common/g2o_types.h"
#include "common/io_utils.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

DEFINE_string(txt_path, "./data/ch3/10.txt", "数据文件路径");
DEFINE_double(antenna_angle, 12.06, "RTK天线安装偏角（角度）");
DEFINE_double(antenna_pox_x, -0.17, "RTK天线安装偏移X");
DEFINE_double(antenna_pox_y, -0.20, "RTK天线安装偏移Y");
DEFINE_bool(with_ui, true, "是否显示图形界面");

TEST(PREINTEGRATION_TEST, ROTATION_TEST) {
    // 测试在恒定角速度运转下的预积分情况
    double imu_time_span = 0.01;       // IMU测量间隔
    Vec3d constant_omega(0, 0, M_PI);  // 角速度为180度/s，转1秒应该等于转180度
    Vec3d gravity(0, 0, -9.8);         // Z 向上，重力方向为负

    sad::NavStated start_status(0), end_status(1.0);
    sad::IMUPreintegration pre_integ;

    // 对比直接积分
    Sophus::SO3d R;
    Vec3d t = Vec3d::Zero();
    Vec3d v = Vec3d::Zero();

    for (int i = 1; i <= 100; ++i) {
        double time = imu_time_span * i;
        Vec3d acce = -gravity;  // 加速度计应该测量到一个向上的力
        pre_integ.Integrate(sad::IMU(time, constant_omega, acce), imu_time_span);

        sad::NavStated this_status = pre_integ.Predict(start_status, gravity);

        t = t + v * imu_time_span + 0.5 * gravity * imu_time_span * imu_time_span +
            0.5 * (R * acce) * imu_time_span * imu_time_span;
        v = v + gravity * imu_time_span + (R * acce) * imu_time_span;
        R = R * Sophus::SO3d::exp(constant_omega * imu_time_span);

        // 验证在简单情况下，直接积分和预积分结果相等
        EXPECT_NEAR(t[0], this_status.p_[0], 1e-2);
        EXPECT_NEAR(t[1], this_status.p_[1], 1e-2);
        EXPECT_NEAR(t[2], this_status.p_[2], 1e-2);

        EXPECT_NEAR(v[0], this_status.v_[0], 1e-2);
        EXPECT_NEAR(v[1], this_status.v_[1], 1e-2);
        EXPECT_NEAR(v[2], this_status.v_[2], 1e-2);

        EXPECT_NEAR(R.unit_quaternion().x(), this_status.R_.unit_quaternion().x(), 1e-4);
        EXPECT_NEAR(R.unit_quaternion().y(), this_status.R_.unit_quaternion().y(), 1e-4);
        EXPECT_NEAR(R.unit_quaternion().z(), this_status.R_.unit_quaternion().z(), 1e-4);
        EXPECT_NEAR(R.unit_quaternion().w(), this_status.R_.unit_quaternion().w(), 1e-4);
    }

    end_status = pre_integ.Predict(start_status);

    spdlog::info("preinteg result: ");
    //
    //
    //
    spdlog::info("end rotation: \n{}", end_status.R_.matrix());
    spdlog::info("end trans: \n{}", end_status.p_.transpose());
    spdlog::info("end v: \n{}", end_status.v_.transpose());

    spdlog::info("direct integ result: ");
    //
    //
    //
    spdlog::info("end rotation: \n{}", R.matrix());
    spdlog::info("end trans: \n{}", t.transpose());
    spdlog::info("end v: \n{}", v.transpose());
    SUCCEED();
}

TEST(PREINTEGRATION_TEST, ACCELERATION_TEST) {
    // 测试在恒定加速度运行下的预积分情况
    double imu_time_span = 0.01;     // IMU测量间隔
    Vec3d gravity(0, 0, -9.8);       // Z 向上，重力方向为负
    Vec3d constant_acce(0.1, 0, 0);  // x 方向上的恒定加速度

    sad::NavStated start_status(0), end_status(1.0);
    sad::IMUPreintegration pre_integ;

    // 对比直接积分
    Sophus::SO3d R;
    Vec3d t = Vec3d::Zero();
    Vec3d v = Vec3d::Zero();

    for (int i = 1; i <= 100; ++i) {
        double time = imu_time_span * i;
        Vec3d acce = constant_acce - gravity;
        pre_integ.Integrate(sad::IMU(time, Vec3d::Zero(), acce), imu_time_span);
        sad::NavStated this_status = pre_integ.Predict(start_status, gravity);

        t = t + v * imu_time_span + 0.5 * gravity * imu_time_span * imu_time_span +
            0.5 * (R * acce) * imu_time_span * imu_time_span;
        v = v + gravity * imu_time_span + (R * acce) * imu_time_span;

        // 验证在简单情况下，直接积分和预积分结果相等
        EXPECT_NEAR(t[0], this_status.p_[0], 1e-2);
        EXPECT_NEAR(t[1], this_status.p_[1], 1e-2);
        EXPECT_NEAR(t[2], this_status.p_[2], 1e-2);

        EXPECT_NEAR(v[0], this_status.v_[0], 1e-2);
        EXPECT_NEAR(v[1], this_status.v_[1], 1e-2);
        EXPECT_NEAR(v[2], this_status.v_[2], 1e-2);

        EXPECT_NEAR(R.unit_quaternion().x(), this_status.R_.unit_quaternion().x(), 1e-4);
        EXPECT_NEAR(R.unit_quaternion().y(), this_status.R_.unit_quaternion().y(), 1e-4);
        EXPECT_NEAR(R.unit_quaternion().z(), this_status.R_.unit_quaternion().z(), 1e-4);
        EXPECT_NEAR(R.unit_quaternion().w(), this_status.R_.unit_quaternion().w(), 1e-4);
    }

    end_status = pre_integ.Predict(start_status);
    spdlog::info("preinteg result: ");
    //
    //
    //
    spdlog::info("end rotation: \n{}", end_status.R_.matrix());
    spdlog::info("end trans: \n{}", end_status.p_.transpose());
    spdlog::info("end v: \n{}", end_status.v_.transpose());

    spdlog::info("direct integ result: ");
    //
    //
    //
    spdlog::info("end rotation: \n{}", R.matrix());
    spdlog::info("end trans: \n{}", t.transpose());
    spdlog::info("end v: \n{}", v.transpose());
    SUCCEED();
}

void Optimize(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& preinteg, const Vec3d& grav);

/// 使用ESKF的Predict, Update来验证预积分的优化过程
TEST(PREINTEGRATION_TEST, ESKF_TEST) {
    if (fLS::FLAGS_txt_path.empty()) {
        FAIL();
    }

    // 初始化器
    sad::StaticIMUInit imu_init;  // 使用默认配置
    sad::ESKFD eskf;

    sad::TxtIO io(FLAGS_txt_path);
    Vec2d antenna_pos(FLAGS_antenna_pox_x, FLAGS_antenna_pox_y);

    std::ofstream fout("./data/ch3/gins.txt");
    bool imu_inited = false, gnss_inited = false;

    /// 设置各类回调函数
    bool first_gnss_set = false;
    Vec3d origin = Vec3d::Zero();

    std::shared_ptr<sad::IMUPreintegration> preinteg = nullptr;

    sad::NavStated last_state;
    bool last_state_set = false;

    sad::GNSS last_gnss;
    bool last_gnss_set = false;

    io.SetIMUProcessFunc([&](const sad::IMU& imu) {
          /// IMU 处理函数
          if (!imu_init.InitSuccess()) {
              imu_init.AddIMU(imu);
              return;
          }

          /// 需要IMU初始化
          if (!imu_inited) {
              // 读取初始零偏，设置ESKF
              sad::ESKFD::Options options;
              // 噪声由初始化器估计
              options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
              options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
              eskf.SetInitialConditions(options, imu_init.GetInitBg(), imu_init.GetInitBa(), imu_init.GetGravity());

              imu_inited = true;
              return;
          }

          if (!gnss_inited) {
              /// 等待有效的RTK数据
              return;
          }

          /// GNSS 也接收到之后，再开始进行预测
          double current_time = eskf.GetNominalState().timestamp_;
          eskf.Predict(imu);

          if (preinteg) {
              preinteg->Integrate(imu, imu.timestamp_ - current_time);

              if (last_state_set) {
                  auto pred_of_preinteg = preinteg->Predict(last_state, eskf.GetGravity());
                  auto pred_of_eskf = eskf.GetNominalState();

                  /// 这两个预测值的误差应该非常接近
                  EXPECT_NEAR((pred_of_preinteg.p_ - pred_of_eskf.p_).norm(), 0, 1e-2);
                  EXPECT_NEAR((pred_of_preinteg.R_.inverse() * pred_of_eskf.R_).log().norm(), 0, 1e-2);
                  EXPECT_NEAR((pred_of_preinteg.v_ - pred_of_eskf.v_).norm(), 0, 1e-2);
              }
          }
      })
        .SetGNSSProcessFunc([&](const sad::GNSS& gnss) {
            /// GNSS 处理函数
            if (!imu_inited) {
                return;
            }

            sad::GNSS gnss_convert = gnss;
            if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos, FLAGS_antenna_angle) || !gnss_convert.heading_valid_) {
                return;
            }

            /// 去掉原点
            if (!first_gnss_set) {
                origin = gnss_convert.utm_pose_.translation();
                first_gnss_set = true;
            }
            gnss_convert.utm_pose_.translation() -= origin;

            // 要求RTK heading有效，才能合入EKF
            auto state_bef_update = eskf.GetNominalState();

            eskf.ObserveGps(gnss_convert);

            // 验证优化过程是否正确
            if (last_state_set && last_gnss_set) {
                auto update_state = eskf.GetNominalState();

                //
                spdlog::info("state after  eskf update: {}", update_state);
                //
                spdlog::info("last state: {}", last_state);

                auto state_pred = preinteg->Predict(last_state, eskf.GetGravity());
                //
                spdlog::info("state in pred: {}", state_pred);

                Optimize(last_state, update_state, last_gnss, gnss_convert, preinteg, eskf.GetGravity());
            }

            last_state = eskf.GetNominalState();
            last_state_set = true;

            // 重置预积分
            sad::IMUPreintegration::Options options;
            options.init_bg_ = last_state.bg_;
            options.init_ba_ = last_state.ba_;
            preinteg = std::make_shared<sad::IMUPreintegration>(options);

            gnss_inited = true;
            last_gnss = gnss_convert;
            last_gnss_set = true;
        })
        .SetOdomProcessFunc([&](const sad::Odom& odom) { imu_init.AddOdom(odom); })
        .Go();

    SUCCEED();
}

void Optimize(sad::NavStated& last_state, sad::NavStated& this_state, sad::GNSS& last_gnss, sad::GNSS& this_gnss,
              std::shared_ptr<sad::IMUPreintegration>& pre_integ, const Vec3d& grav) {
    assert(pre_integ != nullptr);

    if (pre_integ->dt_ < 1e-3) {
        // 未得到积分
        return;
    }

    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto* solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 上时刻顶点， pose, v, bg, ba
    auto v0_pose = new sad::VertexPose();
    v0_pose->setId(0);
    v0_pose->setEstimate(last_state.GetSE3());
    optimizer.addVertex(v0_pose);

    auto v0_vel = new sad::VertexVelocity();
    v0_vel->setId(1);
    v0_vel->setEstimate(last_state.v_);
    optimizer.addVertex(v0_vel);

    auto v0_bg = new sad::VertexGyroBias();
    v0_bg->setId(2);
    v0_bg->setEstimate(last_state.bg_);
    optimizer.addVertex(v0_bg);

    auto v0_ba = new sad::VertexAccBias();
    v0_ba->setId(3);
    v0_ba->setEstimate(last_state.ba_);
    optimizer.addVertex(v0_ba);

    // 本时刻顶点，pose, v, bg, ba
    auto v1_pose = new sad::VertexPose();
    v1_pose->setId(4);
    v1_pose->setEstimate(this_state.GetSE3());
    optimizer.addVertex(v1_pose);

    auto v1_vel = new sad::VertexVelocity();
    v1_vel->setId(5);
    v1_vel->setEstimate(this_state.v_);
    optimizer.addVertex(v1_vel);

    auto v1_bg = new sad::VertexGyroBias();
    v1_bg->setId(6);
    v1_bg->setEstimate(this_state.bg_);
    optimizer.addVertex(v1_bg);

    auto v1_ba = new sad::VertexAccBias();
    v1_ba->setId(7);
    v1_ba->setEstimate(this_state.ba_);
    optimizer.addVertex(v1_ba);

    // 预积分边
    auto edge_inertial = new sad::EdgeInertial(pre_integ, grav);
    edge_inertial->setVertex(0, v0_pose);
    edge_inertial->setVertex(1, v0_vel);
    edge_inertial->setVertex(2, v0_bg);
    edge_inertial->setVertex(3, v0_ba);
    edge_inertial->setVertex(4, v1_pose);
    edge_inertial->setVertex(5, v1_vel);

    auto* rk = new g2o::RobustKernelHuber();
    rk->setDelta(200.0);
    edge_inertial->setRobustKernel(rk);

    optimizer.addEdge(edge_inertial);
    edge_inertial->computeError();
    //
    spdlog::info("inertial init err: {}", edge_inertial->chi2());

    auto* edge_gyro_rw = new sad::EdgeGyroRW();
    edge_gyro_rw->setVertex(0, v0_bg);
    edge_gyro_rw->setVertex(1, v1_bg);
    edge_gyro_rw->setInformation(Mat3d::Identity() * 1e6);
    optimizer.addEdge(edge_gyro_rw);

    edge_gyro_rw->computeError();
    //
    spdlog::info("inertial bg rw: {}", edge_gyro_rw->chi2());

    auto* edge_acc_rw = new sad::EdgeAccRW();
    edge_acc_rw->setVertex(0, v0_ba);
    edge_acc_rw->setVertex(1, v1_ba);
    edge_acc_rw->setInformation(Mat3d::Identity() * 1e6);
    optimizer.addEdge(edge_acc_rw);

    edge_acc_rw->computeError();
    //
    spdlog::info("inertial ba rw: {}", edge_acc_rw->chi2());

    // GNSS边
    auto edge_gnss0 = new sad::EdgeGNSS(v0_pose, last_gnss.utm_pose_);
    edge_gnss0->setInformation(Mat6d::Identity() * 1e2);
    optimizer.addEdge(edge_gnss0);

    edge_gnss0->computeError();
    //
    spdlog::info("gnss0 init err: {}", edge_gnss0->chi2());

    auto edge_gnss1 = new sad::EdgeGNSS(v1_pose, this_gnss.utm_pose_);
    edge_gnss1->setInformation(Mat6d::Identity() * 1e2);
    optimizer.addEdge(edge_gnss1);

    edge_gnss1->computeError();
    //
    spdlog::info("gnss1 init err: {}", edge_gnss1->chi2());

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    sad::NavStated corr_state(this_state.timestamp_, v1_pose->estimate().so3(), v1_pose->estimate().translation(),
                              v1_vel->estimate(), v1_bg->estimate(), v1_ba->estimate());
    //
    spdlog::info("corr state in opt: {}", corr_state);

    // 获取结果，统计各类误差
    spdlog::info("chi2/error: ");
    //
    //
    //
    //
    spdlog::info("preintegration: {}/{}", edge_inertial->chi2(), edge_inertial->error().transpose());
    spdlog::info("gnss0: {}, {}", edge_gnss0->chi2(), edge_gnss0->error().transpose());
    spdlog::info("gnss1: {}, {}", edge_gnss1->chi2(), edge_gnss1->error().transpose());
    spdlog::info("bias: {}/{}", edge_gyro_rw->chi2(), edge_acc_rw->error().transpose());
}

struct InertialData {
    double dt;
    Vec3d gravity;
    SE3 pi;
    Vec3d vi;
    Vec3d bgi;
    Vec3d bai;
    SE3 pj;
    Vec3d vj;
};

class EdgeInertialNumeric : public sad::EdgeInertial {
   public:
    using sad::EdgeInertial::EdgeInertial;
    // force to use numeric jacobian
    void linearizeOplus() override { g2o::BaseMultiEdge<9, Vec9d>::linearizeOplus(); }
};

TEST(PREINTEGRATION_TEST, INERTIAL_JACOBIAN_TEST) {
    using namespace sad;
    {
        // clang-format off
// edge_inertial: dt=0.10736346244812012, gravity=[0,0,-9.8], p1=[-48.20425923233071,131.09441268357259,-0.17555616143992245,0.0028793839792342945,-0.0024040386346471316,-0.013927729053264157,0.9998959686435196], v1=[   1.06207 -0.0706172  0.0273412], bg1=[-0.000472649   0.00017257   0.00118948], ba1=[-0.0953851  0.0117037  0.0314766], p2=[-48.08847834851701,131.0855464835406,-0.17313973584212922,0.002914867414726511,-0.0029468344350995716,-0.013720745004773289,0.999897275159444], v2=[   1.10102 -0.0844701  0.0218921]
// edge_inertial: dt=0.10748577117919922, gravity=[0,0,-9.8], p1=[-42.55171724823842,131.54022416603542,-0.24516467957873317,0.002180353159957587,-0.003142283703716957,-0.008012174931582875,0.999960587806384], v1=[   1.30707 0.00924178  0.0133635], bg1=[-0.000462205  0.000182757   0.00114455], ba1=[-0.0945269  0.0113947   0.028877], p2=[-42.410383736111775,131.53984371451153,-0.24500657858788788,0.00272818337405183,-0.0036145062230732395,-0.008044164478200125,0.9999573909812801], v2=[   1.31712 -0.0201331 0.00505722]
// edge_inertial: dt=0.1089787483215332, gravity=[0,0,-9.8], p1=[27.6904024647889,52.42532709923084,-0.5131977515154368,-0.0018508176696660243,-0.004696131165904422,-0.2250795965667553,0.9743273043671582], v1=[   1.0926 -0.558354 0.0171523], bg1=[-0.000688562  0.000156823   0.00176622], ba1=[ -0.104772 0.00571975  0.0287812], p2=[27.809148629325342,52.364532223849864,-0.5110961726042723,-0.0019540646605160582,-0.004729658462106072,-0.22544471300500388,0.9742425228556931], v2=[  1.08471 -0.558338 0.0145048]
// edge_inertial: dt=0.10906529426574707, gravity=[0,0,-9.8], p1=[30.435419407775125,49.08906027001082,-0.4845537082062321,-3.691353711578972e-05,-0.006187209252838834,-0.221490574926326,0.9751428830165649], v1=[    0.73211   -0.319747 -0.00359525], bg1=[-0.000701906  0.000157548   0.00200918], ba1=[ -0.104825 0.00685935  0.0301422], p2=[30.51473210644913,49.053988712188094,-0.48420151290728264,-0.0007090746684560713,-0.007232142535000457,-0.21809130187953388,0.9759013153859144], v2=[  0.761843  -0.355137 0.00276586]
        // clang-format on
        std::vector<InertialData> inertial_datas;
        inertial_datas.push_back({0.10736346244812012, Vec3d(0, 0, -9.8),
                                  SE3(Eigen::Quaternion<double>(0.0028793839792342945, -0.0024040386346471316,
                                                                -0.013927729053264157, 0.9998959686435196),
                                      Vec3d(-48.20425923233071, 131.09441268357259, -0.17555616143992245)),
                                  Vec3d(1.06207, -0.0706172, 0.0273412), Vec3d(-0.000472649, 0.00017257, 0.00118948),
                                  Vec3d(-0.0953851, 0.0117037, 0.0314766),
                                  SE3(Eigen::Quaternion<double>(0.002914867414726511, -0.0029468344350995716,
                                                                -0.013720745004773289, 0.999897275159444),
                                      Vec3d(-48.08847834851701, 131.0855464835406, -0.17313973584212922)),
                                  Vec3d(1.10102, -0.0844701, 0.0218921)});
        inertial_datas.push_back({0.10748577117919922, Vec3d(0, 0, -9.8),
                                  SE3(Eigen::Quaternion<double>(0.002180353159957587, -0.003142283703716957,
                                                                -0.008012174931582875, 0.999960587806384),
                                      Vec3d(-42.55171724823842, 131.54022416603542, -0.24516467957873317)),
                                  Vec3d(1.30707, 0.00924178, 0.0133635), Vec3d(-0.000462205, 0.000182757, 0.00114455),
                                  Vec3d(-0.0945269, 0.0113947, 0.028877),
                                  SE3(Eigen::Quaternion<double>(0.00272818337405183, -0.0036145062230732395,
                                                                -0.008044164478200125, 0.9999573909812801),
                                      Vec3d(-42.410383736111775, 131.53984371451153, -0.24500657858788788)),
                                  Vec3d(1.31712, -0.0201331, 0.00505722)});
        inertial_datas.push_back({0.1089787483215332, Vec3d(0, 0, -9.8),
                                  SE3(Eigen::Quaternion<double>(-0.0018508176696660243, -0.004696131165904422,
                                                                -0.2250795965667553, 0.9743273043671582),
                                      Vec3d(27.6904024647889, 52.42532709923084, -0.5131977515154368)),
                                  Vec3d(1.0926, -0.558354, 0.0171523), Vec3d(-0.000688562, 0.000156823, 0.00176622),
                                  Vec3d(-0.104772, 0.00571975, 0.0287812),
                                  SE3(Eigen::Quaternion<double>(-0.0019540646605160582, -0.004729658462106072,
                                                                -0.22544471300500388, 0.9742425228556931),
                                      Vec3d(27.809148629325342, 52.364532223849864, -0.5110961726042723)),
                                  Vec3d(1.08471, -0.558338, 0.0145048)});
        inertial_datas.push_back({0.10906529426574707, Vec3d(0, 0, -9.8),
                                  SE3(Eigen::Quaternion<double>(-3.691353711578972e-05, -0.006187209252838834,
                                                                -0.221490574926326, 0.9751428830165649),
                                      Vec3d(30.435419407775125, 49.08906027001082, -0.4845537082062321)),
                                  Vec3d(0.73211, -0.319747, -0.00359525), Vec3d(-0.000701906, 0.000157548, 0.00200918),
                                  Vec3d(-0.104825, 0.00685935, 0.0301422),
                                  SE3(Eigen::Quaternion<double>(-0.0007090746684560713, -0.007232142535000457,
                                                                -0.21809130187953388, 0.9759013153859144),
                                      Vec3d(30.51473210644913, 49.053988712188094, -0.48420151290728264)),
                                  Vec3d(0.761843, -0.355137, 0.00276586)});

        for (const auto& data : inertial_datas) {
            using BlockSolverType = g2o::BlockSolverX;
            using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

            auto imu_pre = std::make_shared<IMUPreintegration>();
            imu_pre->dt_ = data.dt;
            auto get_hessian = [&](sad::EdgeInertial* edge_inertial) {
                auto* solver = new g2o::OptimizationAlgorithmLevenberg(
                    std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
                g2o::SparseOptimizer optimizer;
                optimizer.setAlgorithm(solver);

                auto vi_p = new VertexPose();
                vi_p->setId(0);
                vi_p->setEstimate(data.pi);
                optimizer.addVertex(vi_p);

                auto vi_v = new VertexVelocity();
                vi_v->setId(1);
                vi_v->setEstimate(data.vi);
                optimizer.addVertex(vi_v);

                auto vi_bg = new VertexGyroBias();
                vi_bg->setId(2);
                vi_bg->setEstimate(data.bgi);
                optimizer.addVertex(vi_bg);

                auto vi_ba = new VertexAccBias();
                vi_ba->setId(3);
                vi_ba->setEstimate(data.bai);
                optimizer.addVertex(vi_ba);

                auto vj_p = new VertexPose();
                vj_p->setId(4);
                vj_p->setEstimate(data.pj);
                optimizer.addVertex(vj_p);

                auto vj_v = new VertexVelocity();
                vj_v->setId(5);
                vj_v->setEstimate(data.vj);
                optimizer.addVertex(vj_v);

                edge_inertial->setInformation(Mat9d::Identity());
                edge_inertial->setVertex(0, vi_p);
                edge_inertial->setVertex(1, vi_v);
                edge_inertial->setVertex(2, vi_bg);
                edge_inertial->setVertex(3, vi_ba);
                edge_inertial->setVertex(4, vj_p);
                edge_inertial->setVertex(5, vj_v);
                optimizer.addEdge(edge_inertial);
                optimizer.initializeOptimization();
                optimizer.optimize(1);
                return edge_inertial->GetHessian();
            };
            auto H_analysis = get_hessian(new sad::EdgeInertial(imu_pre, data.gravity));
            auto H_numeric = get_hessian(new EdgeInertialNumeric(imu_pre, data.gravity));
            EXPECT_TRUE(H_analysis.isApprox(H_numeric, 1e-5));
        }
    }
}

struct WheelData {
    SE3 pj;
    Vec3d vj;
};

class EdgeWheelSpeedNumeric : public sad::EdgeWheelSpeed {
   public:
    using sad::EdgeWheelSpeed::EdgeWheelSpeed;
    // force to use numeric jacobian
    void linearizeOplus() override {
        g2o::BaseBinaryEdge<3, Vec3d, sad::VertexVelocity, sad::VertexPose>::linearizeOplus();
    }
};

TEST(PREINTEGRATION_TEST, WHEEL_JACOBIAN_TEST) {
    using namespace sad;
    {
        // clang-format off
// edge_wheel: p2=[-121.71375069052047,1.6693158795978535,-0.630749876343679,0.010943148703749324,-0.0057734802704026795,0.08483334000128599,0.9963183320837979], v2=[  1.26815  0.318286 0.0161879]
// edge_wheel: p2=[-75.79195832238703,25.06890381364548,-0.6090009027177773,0.007936563191486688,-0.00012309451614146724,0.7102719890375541,0.7038825877950702], v2=[-0.00309099     1.20132   0.0188821]
// edge_wheel: p2=[29.21454427951416,101.08244311950648,-0.3631043579593454,0.006255023556490374,-0.008608298129634226,0.9609693063078633,-0.2764502924899942], v2=[   -1.07971   -0.736306 -0.00938653]
// edge_wheel: p2=[24.56944983936715,65.2221625478426,-0.5140032012727107,-0.0006136493388138473,0.001149997728963421,-0.6048605013057748,-0.7963303804953314], v2=[   0.296722     1.27953 0.000438097]        // clang-format on
        std::vector<WheelData> datas;
        datas.push_back({SE3(Eigen::Quaternion<double>(0.010943148703749324, -0.0057734802704026795,
                                                       0.08483334000128599, 0.9963183320837979),
                             Vec3d(-121.71375069052047, 1.6693158795978535, -0.630749876343679)),
                         Vec3d(1.26815, 0.318286, 0.0161879)});
        datas.push_back({SE3(Eigen::Quaternion<double>(0.007936563191486688, -0.00012309451614146724,
                                                       0.7102719890375541, 0.7038825877950702),
                             Vec3d(-75.79195832238703, 25.06890381364548, -0.6090009027177773)),
                         Vec3d(-0.00309099, 1.20132, 0.0188821)});
        datas.push_back({SE3(Eigen::Quaternion<double>(0.006255023556490374, -0.008608298129634226,
                                                       0.9609693063078633, -0.2764502924899942),
                             Vec3d(29.21454427951416, 101.08244311950648, -0.3631043579593454)),
                         Vec3d(-1.07971, -0.736306, -0.00938653)});
        datas.push_back({SE3(Eigen::Quaternion<double>(-0.0006136493388138473, 0.001149997728963421,
                                                       -0.6048605013057748, -0.7963303804953314),
                             Vec3d(24.56944983936715, 65.2221625478426, -0.5140032012727107)),
                         Vec3d(0.296722, 1.27953, 0.000438097)});
        for (const auto& data : datas) {
            using BlockSolverType = g2o::BlockSolverX;
            using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;


            auto get_hessian = [&](auto *edge_wheel) {
                auto* solver = new g2o::OptimizationAlgorithmLevenberg(
                    std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
                g2o::SparseOptimizer optimizer;
                optimizer.setAlgorithm(solver);

                auto vj_p = new VertexPose();
                vj_p->setId(0);
                vj_p->setEstimate(data.pj);
                optimizer.addVertex(vj_p);

                auto vj_v = new VertexVelocity();
                vj_v->setId(1);
                vj_v->setEstimate(data.vj);
                optimizer.addVertex(vj_v);

                edge_wheel = new std::remove_pointer_t<decltype(edge_wheel)>(vj_v, vj_p,
                                                          Vec3d(0.0, 0.0, 0.0));
                edge_wheel->setInformation(Mat3d::Identity());
                optimizer.addEdge(edge_wheel);
                optimizer.initializeOptimization();
                optimizer.optimize(1);
                return edge_wheel->GetHessian();
            };
            auto H_analysis = get_hessian((sad::EdgeWheelSpeed *)nullptr);
            auto H_numeric = get_hessian((EdgeWheelSpeedNumeric *)nullptr);
            EXPECT_TRUE(H_analysis.isApprox(H_numeric, 1e-5));
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
