get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(${LIB_NAME} bfnn.cc kdtree.cc octo_tree.cc)
target_link_libraries(${LIB_NAME} PUBLIC ${PROJECT_NAME}.common)

set(TARGET_NAME ${DIR_NAME}.point_cloud_load_and_vis)
add_executable(${TARGET_NAME} point_cloud_load_and_vis.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})

set(TARGET_NAME ${DIR_NAME}.pcd_to_bird_eye)
add_executable(${TARGET_NAME} pcd_to_bird_eye.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})

set(TARGET_NAME ${DIR_NAME}.scan_to_range_image)
add_executable(${TARGET_NAME} scan_to_range_image.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})

set(TARGET_NAME ${DIR_NAME}.linear_fitting)
add_executable(${TARGET_NAME} linear_fitting.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})

set(TARGET_NAME ${DIR_NAME}.test_nn)
add_executable(${TARGET_NAME} test_nn.cc)
target_link_libraries(${TARGET_NAME} GTest::gtest ${LIB_NAME})
add_test(${TARGET_NAME} ${TARGET_NAME})
