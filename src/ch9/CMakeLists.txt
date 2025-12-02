get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(${LIB_NAME} frontend.cc keyframe.cc optimization.cc loopclosure.cc)

target_link_libraries(
  ${LIB_NAME} PUBLIC ${PROJECT_NAME}.ch8 ${PROJECT_NAME}.ch4
                     ${PROJECT_NAME}.ch3)

function(add_exec src_name)
  set(TARGET_NAME ${DIR_NAME}.${src_name})
  add_executable(${TARGET_NAME} ${src_name}.cc)
  target_link_libraries(${TARGET_NAME} ${LIB_NAME})
endfunction()

add_exec(run_frontend)
add_exec(run_optimization)
add_exec(run_loopclosure)
add_exec(run_mapping)
add_exec(dump_map)
add_exec(split_map)
