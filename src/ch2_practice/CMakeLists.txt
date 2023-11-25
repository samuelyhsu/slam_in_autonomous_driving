set(TARGET_NAME motion_p)
add_executable(${TARGET_NAME} motion.cc)
target_link_libraries(${TARGET_NAME}
        ${PROJECT_NAME}.common
        ${PROJECT_NAME}.tools
        )
