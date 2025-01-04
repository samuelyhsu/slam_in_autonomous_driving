add_library(${PROJECT_NAME}.ch5
        bfnn.cc
        kdtree.cc
        octo_tree.cc
        )
target_link_libraries(${PROJECT_NAME}.ch5 PUBLIC
${PROJECT_NAME}.common
        )
        
add_executable(point_cloud_load_and_vis point_cloud_load_and_vis.cc)
target_link_libraries(point_cloud_load_and_vis
${PROJECT_NAME}.ch5
        )

add_executable(pcd_to_bird_eye pcd_to_bird_eye.cc)
target_link_libraries(pcd_to_bird_eye
${PROJECT_NAME}.ch5
        )

add_executable(scan_to_range_image scan_to_range_image.cc)
target_link_libraries(scan_to_range_image
${PROJECT_NAME}.ch5
        )



add_executable(linear_fitting linear_fitting.cc)
target_link_libraries(linear_fitting
${PROJECT_NAME}.ch5
        )

ADD_EXECUTABLE(test_nn test_nn.cc)
ADD_TEST(test_nn test_bfnn)

target_link_libraries(test_nn
        GTest::gtest ${PROJECT_NAME}.ch5
        )
