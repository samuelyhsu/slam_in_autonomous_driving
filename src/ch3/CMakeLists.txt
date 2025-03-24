get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(
  ${LIB_NAME}
  static_imu_init.cc utm_convert.cc
  ${PROJECT_SOURCE_DIR}/thirdparty/utm_convert/utm.cc
  ${PROJECT_SOURCE_DIR}/thirdparty/utm_convert/tranmerc.cc)
# target_compile_definitions(${LIB_NAME} PUBLIC USE_LEFT_PERTURBATION)
# target_compile_definitions(${LIB_NAME} PUBLIC USE_ORIGINAL_CODE)
target_link_libraries(${LIB_NAME} PUBLIC ${PROJECT_NAME}.common)

set(TARGET_NAME ${DIR_NAME}.run_imu_integration)
add_executable(${TARGET_NAME} run_imu_integration.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})

set(TARGET_NAME ${DIR_NAME}.run_eskf_gins)
add_executable(${TARGET_NAME} run_eskf_gins.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})

set(TARGET_NAME ${DIR_NAME}.process_gnss)
add_executable(${TARGET_NAME} process_gnss.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})
