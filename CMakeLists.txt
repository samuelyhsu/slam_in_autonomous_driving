cmake_minimum_required(VERSION 3.5)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
endif()

list(APPEND CMAKE_PREFIX_PATH "/opt/ros/noetic")

message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

project(slam_in_auto_driving)

set(CMAKE_CXX_STANDARD 17)

add_definitions("-DCATKIN_ENABLE_TESTING=0")

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(thirdparty/livox_ros_driver)

function(print_package_info package_name)
  message(
    STATUS
      "${package_name}: version=${${package_name}_VERSION}, dir=${${package_name}_DIR}"
  )
endfunction()

find_package(spdlog CONFIG REQUIRED)
print_package_info(spdlog)

find_package(GTest CONFIG REQUIRED)
print_package_info(GTest)

find_package(gflags CONFIG REQUIRED)
print_package_info(gflags)

find_package(TBB CONFIG REQUIRED)
print_package_info(TBB)

find_package(Eigen3 CONFIG REQUIRED)
print_package_info(Eigen3)

find_package(Sophus REQUIRED)
print_package_info(Sophus)

find_package(CHOLMOD CONFIG REQUIRED)
print_package_info(CHOLMOD)

find_package(CXSparse CONFIG REQUIRED)
print_package_info(CXSparse)

find_package(g2o REQUIRED)
print_package_info(g2o)
set(g2o_libs
    g2o::stuff
    g2o::core
    g2o::types_sba
    g2o::types_icp
    g2o::solver_dense
    g2o::solver_csparse
    g2o::csparse_extension
    g2o::solver_cholmod
    g2o::solver_eigen)

find_package(Pangolin CONFIG REQUIRED)
print_package_info(Pangolin)

find_package(yaml-cpp CONFIG REQUIRED)
print_package_info(yaml-cpp)

find_package(PCL CONFIG REQUIRED)
print_package_info(PCL)

find_package(OpenCV CONFIG REQUIRED)
print_package_info(OpenCV)

include_directories(${CMAKE_BINARY_DIR}/devel/include)

find_package(
  catkin REQUIRED
  COMPONENTS roscpp
             rospy
             std_msgs
             sensor_msgs
             geometry_msgs
             message_generation
             pcl_ros
             pcl_conversions)
include_directories(${catkin_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/thirdparty/)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/velodyne/include)

set(third_party_libs
    # from vcpkg
    spdlog::spdlog
    Eigen3::Eigen
    Sophus::Sophus
    ${Pangolin_LIBRARIES}
    SuiteSparse::CHOLMOD
    SuiteSparse::CXSparse
    yaml-cpp::yaml-cpp
    ${g2o_libs}
    # others
    pthread
    gflags
    TBB::tbb
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES})
message(STATUS "third_party_libs=${third_party_libs}")

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(src)
