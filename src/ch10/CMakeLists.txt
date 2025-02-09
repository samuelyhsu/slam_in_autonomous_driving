add_library(${PROJECT_NAME}.ch10 fusion.cc)

target_link_libraries(${PROJECT_NAME}.ch10 PUBLIC ${PROJECT_NAME}.ch3
                                                  ${PROJECT_NAME}.ch7)

add_executable(run_fusion_offline run_fusion_offline.cc)
target_link_libraries(run_fusion_offline ${PROJECT_NAME}.ch10)
