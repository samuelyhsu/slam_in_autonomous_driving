get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(
  ${LIB_NAME}
  static_imu_init.cc utm_convert.cc
  ${PROJECT_SOURCE_DIR}/thirdparty/utm_convert/utm.cc
  ${PROJECT_SOURCE_DIR}/thirdparty/utm_convert/tranmerc.cc)

target_link_libraries(${LIB_NAME} PUBLIC ${PROJECT_NAME}.common)

set(TARGET_NAME run_imu_integration.${DIR_NAME})
add_executable(${TARGET_NAME} run_imu_integration.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})

set(TARGET_NAME run_eskf_gins.${DIR_NAME})
add_executable(${TARGET_NAME} run_eskf_gins.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})

set(TARGET_NAME process_gnss.${DIR_NAME})
add_executable(${TARGET_NAME} process_gnss.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})
