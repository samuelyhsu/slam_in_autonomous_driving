//
// Created by xiang on 22-12-20.
//

#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "frontend.h"
#include "loopclosure.h"
#include "optimization.h"

DEFINE_string(config_yaml, "./config/mapping.yaml", "配置文件");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    spdlog::info("testing frontend");
    sad::Frontend frontend(FLAGS_config_yaml);
    if (!frontend.Init()) {
        spdlog::error("failed to init frontend.");
        return -1;
    }

    frontend.Run();

    sad::Optimization opti(FLAGS_config_yaml);
    if (!opti.Init(1)) {
        spdlog::error("failed to init opti1.");
        return -1;
    }
    opti.Run();

    sad::LoopClosure lc(FLAGS_config_yaml);
    if (!lc.Init()) {
        spdlog::error("failed to init loop closure.");
        return -1;
    }
    lc.Run();

    sad::Optimization opti2(FLAGS_config_yaml);
    if (!opti2.Init(2)) {
        spdlog::error("failed to init opti2.");
        return -1;
    }
    opti2.Run();

    spdlog::info("done.");
    return 0;
}
