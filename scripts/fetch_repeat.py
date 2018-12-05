
#!/usr/bin/env python

'''
Modified by Allen Spain
12/1/2018
'''

# TurtleBot must have minimal.launch & amcl_demo.launch
# running prior to starting this script
# For simulation: launch gazebo world & amcl_demo prior to run this script

import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion


class GoToPose():
    def __init__(self):

        self.goal_sent = False

        # What to do if shut down (e.g. Ctrl-C or failure)
        rospy.on_shutdown(self.shutdown)

        # Tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient(
            "robot_1/move_base", MoveBaseAction)
        rospy.loginfo("Wait for the action server to come up")

        # Allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))

    def goto(self, pos, quat):
        # Send a goal
        for i in range(len(pos['x'])):
            self.goal_sent = True
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'robot_1/map'
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose = Pose(Point(pos['x'][i], pos['y'][i], 0.000),
                                         Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))

            # print target
            rospy.loginfo(goal.target_pose.pose)

            # Start moving
            self.move_base.send_goal(goal)

            # Allow TurtleBot up to 60 seconds to complete task
            success = self.move_base.wait_for_result(rospy.Duration(160))

            state = self.move_base.get_state()
	    rospy.loginfo("move base state = " + str(state))
	    rospy.loginfo("goal sent = " + str(self.goal_sent))
            result = False

            if success and state == GoalStatus.SUCCEEDED:
                # We made it!
                result = True
            else:
                self.move_base.cancel_goal()
		break

            self.goal_sent = False
        return result

    def shutdown(self):
        if self.goal_sent:
            self.move_base.cancel_goal()
        rospy.loginfo("Stop")
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        rospy.init_node('nav_test', anonymous=False)
        navigator = GoToPose()

        # Customize the following values so they are appropriate for your location

        # position = {'x': 1.36, 'y': -1.46}
        posX = [0, 1.0, 2.0, 3.0]
        posY = [0.0, 0.0, 0.0, 0.0]
        position = {'x': posX, 'y': posY}

        quaternion = {'r1': 0.000, 'r2': 0.000, 'r3': 0.000, 'r4': 1.000}

        # rospy.loginfo("Go to (%s, %s) pose", position['x'], position['y'])
        success = navigator.goto(position, quaternion)

        if success:
            rospy.loginfo("Bingo! reached the desired pose")
        else:
            rospy.loginfo("The base failed to reach the desired position")

        # Sleep to give the last log messages time to be sent
        rospy.sleep(1)

    except rospy.ROSInterruptException:
        rospy.loginfo("Ctrl-C caught. Quitting")

