get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME ${DIR_NAME}.motion)
add_executable(${TARGET_NAME} motion.cc)
target_link_libraries(${TARGET_NAME} ${PROJECT_NAME}.common)
