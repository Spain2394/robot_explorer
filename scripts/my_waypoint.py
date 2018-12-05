#!/usr/bin/env python

from smach import State, StateMachine
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import rospy
from math import *

waypoints = [
    ['one', (0, 1), (0.0, 0.0, 0.0, 1.0)],
    ['two', (2, 5), (0.0, 0.0, -0.984047240305, 0.177907360295)]
]


class Waypoint(State):
    def __init__(self, position, orientation):
        State.__init__(self, outcomes=['success'])

        self.radius = 1
        self.theta = 0

        # Get an action client
        self.client = actionlib.SimpleActionClient(
            'robot_1/move_base', MoveBaseAction)
        self.client.wait_for_server()

        # Define the goal
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'robot_1/map'
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = 0.0
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]

    def execute(self, userdata):
        self.client.send_goal(self.goal)
        self.client.wait_for_result()
        return 'success'


if __name__ == '__main__':
    rospy.init_node('patrol')
    theta = 0
    radius = 1

    patrol = StateMachine('success')
    with patrol:
        for i, w in enumerate(waypoints):
            #w[1] = (w[1][0] + cos(theta + pi/2), w[1][1] + sin(theta + pi/2))
	    rospy.loginfo("w[1]" + str(w[1]))
            StateMachine.add(w[0],
                             Waypoint(w[1], w[2]),
                             transitions={'success': waypoints[(i + 1) %
                                                               len(waypoints)][0]})

    patrol.execute()

