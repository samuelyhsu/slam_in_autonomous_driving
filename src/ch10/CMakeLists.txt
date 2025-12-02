get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(${LIB_NAME} fusion.cc)
target_link_libraries(${LIB_NAME} PUBLIC ${PROJECT_NAME}.ch3
                                         ${PROJECT_NAME}.ch7)

set(TARGET_NAME ${DIR_NAME}.run_fusion_offline)
add_executable(${TARGET_NAME} run_fusion_offline.cc)
target_link_libraries(${TARGET_NAME} ${LIB_NAME})
