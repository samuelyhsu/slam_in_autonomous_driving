//
// Created by xiang on 22-9-9.
//

#include <pcl/io/pcd_io.h>
#include "gflags/gflags.h"
#include "spdlog/spdlog.h"

#include "tools/ui/pangolin_window.h"

DEFINE_string(source, "./data/ch5/first.pcd", "第1个点云路径");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::ui::PangolinWindow ui;
    ui.Init();
    sad::CloudPtr source(new sad::PointCloudType);
    pcl::io::loadPCDFile(fLS::FLAGS_source, *source);

    spdlog::info("set state");
    ui.UpdateScan(source, SE3());

    spdlog::info("waiting");
    sleep(60);
    ui.Quit();

    return 0;
}
