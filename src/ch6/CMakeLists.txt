get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(
  ${LIB_NAME}
  icp_2d.cc
  lidar_2d_utils.cc
  likelihood_field.cc
  occupancy_map.cc
  submap.cc
  mapping_2d.cc
  multi_resolution_likelihood_field.cc
  loop_closing.cc
  frame.cc)
target_link_libraries(${LIB_NAME} PUBLIC ${PROJECT_NAME}.common)

function(add_exec src_name)
  set(TARGET_NAME ${DIR_NAME}.${src_name})
  add_executable(${TARGET_NAME} ${src_name}.cc)
  target_link_libraries(${TARGET_NAME} ${LIB_NAME})
endfunction()

add_exec(test_2dlidar_io)
add_exec(test_2d_icp_s2s)
add_exec(test_2d_icp_likelihood)
add_exec(test_occupancy_grid)
add_exec(test_2d_mapping)
add_exec(test_mr_matching)
