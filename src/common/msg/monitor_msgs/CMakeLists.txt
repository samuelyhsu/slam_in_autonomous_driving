project(monitor_msgs)


add_message_files(
        FILES
        fault_info.msg
        fault_vec.msg
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        std_msgs
)

catkin_package(
        CATKIN_DEPENDS message_runtime
)
