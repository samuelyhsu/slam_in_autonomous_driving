get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(
  ${LIB_NAME}
  ikd_tree/ikd_tree.cpp
  icp_3d.cc
  icp_inc.cpp
  ndt_3d.cc
  ndt_inc.cc
  gen_simu_data.cc
  direct_ndt_lo.cc
  incremental_ndt_lo.cc
  loam-like/feature_extraction.cc
  loam-like/loam_like_odom.cc
  loosely_coupled_lio/loosely_lio.cc
  loosely_coupled_lio/cloud_convert.cc
  loosely_coupled_lio/measure_sync.cc)
target_link_libraries(${LIB_NAME} PUBLIC ${PROJECT_NAME}.ch3
                                         ${PROJECT_NAME}.ch5)

function(add_exec src_name)
  set(TARGET_NAME ${DIR_NAME}.${src_name})
  add_executable(${TARGET_NAME} test/${src_name}.cc)
  target_link_libraries(${TARGET_NAME} ${LIB_NAME})
endfunction()

add_exec(test_icp)
add_exec(test_ndt_lo)
add_exec(test_inc_ndt_lo)
add_exec(test_feature_extraction)
add_exec(test_loam_odom)
add_exec(test_loosely_lio)
add_exec(run_gen_simu_data)
