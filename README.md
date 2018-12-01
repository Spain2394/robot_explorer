## robot-exploration
- Turtlebot RRT exploration and target detection using SIFT

## Setup
- to bringup Turtlebot and perform mapping and localization run ```roslaunch robot_explorer setup.launch```

## Laser Scanner
- to publish to the scan topic using live hokuyo sensor data run: ```rosrun urg_node urg_node```


### Teleoperation
- Should get a response when calling```jstest /dev/input/js0```
- ```roslaunch turtlebot_teleop my_ps3_teleop.launch```

