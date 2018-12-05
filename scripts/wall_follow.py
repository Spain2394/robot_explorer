#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped


class My_TurtleBot():
    def __init__(self):
        rospy.init_node('Move_Turtle', anonymous=False)

        self.pub = rospy.Publisher(
            '/move_base/goal', PoseStamped, queue_size=10)

        while not rospy.is_shutdown():
            for i in range(5):
                goal = PoseStamped()
	        goal.pose.position.x = 1 + i
                goal.pose.position.y = 0
                self.pub.publish(goal)


    def shutdown(self):
        rospy.loginfo("Stop !")
        self.pub.publish(())
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        My_TurtleBot()  # Test your functions
    except rospy.ROSInterruptException:
        pass

