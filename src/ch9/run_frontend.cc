//
// Created by xiang on 22-12-7.
//

#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "frontend.h"

// 测试前端的工作情况

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
    return 0;
}
