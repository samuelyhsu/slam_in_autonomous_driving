//
// Created by xiang on 22-12-19.
//

#include "gflags/gflags.h"
#include "loopclosure.h"
#include "spdlog/spdlog.h"

DEFINE_string(config_yaml, "./config/mapping.yaml", "配置文件");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::LoopClosure lc(FLAGS_config_yaml);
    lc.Init();
    lc.Run();

    return 0;
}
