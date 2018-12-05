## robot-exploration
- Turtlebot RRT exploration and target detection using SIFT

## Requirements
Install packages: </b>
1) Clone RRT exploration package in you ```catkin_ws```
2) Install ROS package navigation stack, for kinetic run
```sudo apt-get install ros-kinetic-navigation```
3) Ensure that you have ROS package gmapping, for kinetic run
```sudo apt-get install ros-kinetic-navigation```
4) clone ```rrt_exploration_tutorials``` for simulation
5) clone ROS package ```rrr_exploration``` for Physical Turtlbot
6) clone ROS package ```urg_node```
7) clone ```robot_explorer``` from [source](https://github.com/Spain2394/robot_explorer)
8) Make workspace with command ```catkin_make``` in your ```~/[catkin_ws]```
9) Source ws by running  ```source devel/setup.bash``` in you  ```catkin_ws```

For more information visit: [RRT wiki](http://wiki.ros.org/rrt_exploration), [Hokuyo Driver wiki](http://wiki.ros.org/urg_node)

## Working with hardware
**Laser Scanner**

- check for usb connectivity by running:```ls -l /dev/ttyACM0```
- to publish to the scan topic using live hokuyo sensor data run: ```rosrun urg_node urg_node```

**Teleoperation**
- Controller config can be evaluated by running: ```jstest /dev/input/js0```
- Create ```my_ps3_teleop.launch``` to reflect controller config
- To test using controller teleop run: ```roslaunch turtlebot_teleop my_ps3_teleop.launch```

------
## Demos
* [Wall Follower](#wall-follower)
* [RRT Path Planning](#rrt-path-planning)
* [SIFT Feature Detector](#sift-feature-detector)

### Wall Follower
Simulates and stages bot in RVIZ and Gazebo, contains wall follower node which subscribes to ```rrt_exploration``` topic ```/robot_1/base_scan```
This also publishes to topic ```/robot_1/mobile_base/commands/velocity``` which drives the bot
* run: ```roslaunch robot_explorer wall_follow.launch```

![1](https://github.com/Spain2394/robot_explorer/blob/master/ScreenCaps/wall_sim.gif)




For more information: [Wall Follower](https://syrotek.felk.cvut.cz/course/ROS_CPP_INTRO/exercise/ROS_CPP_WALLFOLLOWING)


### RRT Path Planning
RRT Path planning using goals provided by service provider ``` fetch_goal.py```. The service provider posts 10 goals each further from the origin than the last. This script can be easily modified to post targets around the map for exploration and target discovery. 
* To run: ```roslaunch rrt_exploration_tutorials single_simulated_house.launch```
* To run service service provider run: ```python fetch_goal.py```
*

![1](https://github.com/Spain2394/robot_explorer/blob/master/ScreenCaps/rrt_sim.gif)

Green line is the robots local plan


### SIFT Feature Detector
* Scale invariant feature detection which takes an image of an object and a target image as input, and outputs a graphical image of the objects location as output, if the object is found.
* To test with image run: ```asfasdf```
*

### to run on TurtleBot
- run: ```roslaunch robot_explorer setup.launch```
- run: ```python fetch_goal.py```

### to run wall follower in simmulation
 - run: ```roslaunch rrt_exploration_tutorials single_simulated_house.launch```
 - run: ```roslaunch robot _explorer wall_follow.launch```

### to run target waypoint simulalation
- run: ``` roslaunch rrt_exploration_tutorials single_simulated_house.launch```
- run: ``` python my_waypoint.py ```
