add_library(${PROJECT_NAME}.ch4 gins_pre_integ.cc imu_preintegration.cc
                                g2o_types.cc)
target_link_libraries(${PROJECT_NAME}.ch4 PUBLIC ${PROJECT_NAME}.common)

add_executable(test_preintegration test_preintegration.cc)
add_test(test_preintegration test_preintegration)

target_link_libraries(test_preintegration GTest::gtest ${PROJECT_NAME}.ch4
                      ${PROJECT_NAME}.ch3)

add_executable(run_gins_pre_integ run_gins_pre_integ.cc)
target_link_libraries(run_gins_pre_integ ${PROJECT_NAME}.ch3
                      ${PROJECT_NAME}.ch4)
