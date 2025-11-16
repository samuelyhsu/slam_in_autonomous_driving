//
// Created by xiang on 22-11-10.
//

#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "ch8/lio-iekf/lio_iekf.h"
#include "common/io_utils.h"
#include "common/sys_utils.h"
#include "common/timer/timer.h"

DEFINE_string(bag_path, "./dataset/sad/nclt/20120115.bag", "path to rosbag");
DEFINE_string(dataset_type, "NCLT", "NCLT/ULHK/UTBM/AVIA");                   // 数据集类型
DEFINE_string(config, "./config/velodyne_nclt.yaml", "path of config yaml");  // 配置文件类型
DEFINE_bool(display_map, true, "display map?");
DEFINE_bool(use_ndt, true, "use NDT for mapping? Set false to use ICP");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path, sad::Str2DatasetType(FLAGS_dataset_type));

    sad::LioIEKF::Options options;
    options.use_ndt_ = FLAGS_use_ndt;
    sad::LioIEKF lio(options);
    lio.Init(FLAGS_config);

    rosbag_io
        .AddAutoPointCloudHandle([&](sensor_msgs::PointCloud2::Ptr cloud) -> bool {
            sad::common::Timer::Evaluate([&]() { lio.PCLCallBack(cloud); }, "IEKF lio");
            return true;
        })
        .AddLivoxHandle([&](const livox_ros_driver::CustomMsg::ConstPtr& msg) -> bool {
            sad::common::Timer::Evaluate([&]() { lio.LivoxPCLCallBack(msg); }, "IEKF lio");
            return true;
        })
        .AddImuHandle([&](IMUPtr imu) {
            lio.IMUCallBack(imu);
            return true;
        })
        .Go();

    lio.Finish();
    sad::common::Timer::PrintAll();
    spdlog::info("done.");

    return 0;
}
