//
// Created by xiang on 22-12-7.
//

#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "optimization.h"

// 测试优化的工作情况

DEFINE_string(config_yaml, "./config/mapping.yaml", "配置文件");
DEFINE_int64(stage, 1, "运行第1阶段或第2阶段优化");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    assert(FLAGS_stage == 1 || FLAGS_stage == 2);

    spdlog::info("testing optimization");
    sad::Optimization opti(FLAGS_config_yaml);
    if (!opti.Init(FLAGS_stage)) {
        spdlog::error("failed to init frontend.");
        return -1;
    }

    opti.Run();
    return 0;
}
