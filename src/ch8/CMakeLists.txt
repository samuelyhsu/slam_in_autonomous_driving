get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME ${PROJECT_NAME}.${DIR_NAME})
add_library(${LIB_NAME} lio-iekf/lio_iekf.cc lio-preinteg/lio_preinteg.cc)

target_link_libraries(
  ${LIB_NAME} PUBLIC ${PROJECT_NAME}.ch7 ${PROJECT_NAME}.ch3
                     ${PROJECT_NAME}.ch4)

function(add_exec src_name)
  set(TARGET_NAME ${DIR_NAME}.${src_name})
  add_executable(${TARGET_NAME} test/${src_name}.cc)
  target_link_libraries(${TARGET_NAME} ${LIB_NAME})
endfunction()

add_exec(test_lio_iekf)
add_exec(test_lio_preinteg)
