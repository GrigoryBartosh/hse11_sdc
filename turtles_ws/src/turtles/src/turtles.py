#! /usr/bin/python2

import rospy

from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

import time

import numpy as np


class TurtleHunter:
    def __init__(self):
        self.publisher = rospy.Publisher('/raphael/cmd_vel', Twist, queue_size=10)
        self.subscriber_me = rospy.Subscriber('/raphael/pose', Pose, self.callback_me)
        self.subscriber_target = rospy.Subscriber('/turtle1/pose', Pose, self.callback_target)

        self.my_pose = None

    def callback_me(self, my_pose):
        self.my_pose = my_pose

    def callback_target(self, target_pose):
        if self.my_pose is None:
            return

        dx = target_pose.x - self.my_pose.x
        dy = target_pose.y - self.my_pose.y
        
        msg = Twist()
        msg.linear.x = dx
        msg.linear.y = dy

        self.publisher.publish(msg)


rospy.init_node('turtle_hunter')
turtle_hunter = TurtleHunter()
rospy.spin()
