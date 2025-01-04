project(velodyne_msgs)

find_package(catkin REQUIRED COMPONENTS monitor_msgs)

add_message_files(
        DIRECTORY msg
        FILES
        VelodynePacket.msg
        VelodyneScan.msg
        VelodyneScanRaw.msg
)
generate_messages(DEPENDENCIES std_msgs monitor_msgs)
catkin_package(
        CATKIN_DEPENDS message_runtime std_msgs monitor_msgs
)
