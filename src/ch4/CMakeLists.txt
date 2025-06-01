get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(${LIB_NAME} gins_pre_integ.cc imu_preintegration.cc g2o_types.cc)
target_link_libraries(${LIB_NAME} PUBLIC ${PROJECT_NAME}.common)

set(TARGET_NAME ${DIR_NAME}.test_preintegration)
add_executable(${TARGET_NAME} test_preintegration.cc)
add_test(${TARGET_NAME} ${TARGET_NAME})
target_link_libraries(${TARGET_NAME} GTest::gtest ${LIB_NAME}
                      ${PROJECT_NAME}.ch3)

set(TARGET_NAME ${DIR_NAME}.run_gins_pre_integ)
add_executable(${TARGET_NAME} run_gins_pre_integ.cc)
target_link_libraries(${TARGET_NAME} ${PROJECT_NAME}.ch3 ${LIB_NAME})
