#! /usr/bin/python2

import numpy as np

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid


rospy.init_node('laser_filter')


class LaserFilter:
    def __init__(self):
        self.map_publisher = rospy.Publisher('/map_topic', OccupancyGrid, queue_size=10)
        self.rate = rospy.Rate(10)

    def msg2polar(self, msg):
        r = np.array(msg.ranges)
        th = msg.angle_min + msg.angle_increment * np.arange(r.shape[0])
        return r, th

    def polar2cartesian(self, r, th):
        x = r * np.cos(th)
        y = r * np.sin(th)
        return np.stack([x, y], axis=1)

    def filter_r(self, r, th):
        ids = np.where(r > 0.5)[0]
        return r[ids], th[ids]

    def filter_alpha(self, p):
        ALPHA = np.pi / 2

        v1 = p[:-2] - p[1:-1]
        v2 = p[2:] - p[1:-1]

        v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

        cos_a = (v1 * v2).sum(axis=1)

        a = np.arccos(cos_a)
        a = np.where(a > np.pi, a - 2 * np.pi, a)

        a = a[1:-1]
        ids = np.where(np.abs(a) > ALPHA)[0]
        ids = ids + 1

        return p[ids]

    def make_grid(self, p):
        RES = 0.05
        N = 200
        R = RES * N / 2

        grid = OccupancyGrid()

        grid.header.frame_id = 'base_laser_link'

        grid.info.resolution = RES
        grid.info.width = N
        grid.info.height = N

        grid.info.origin.position.x = -R
        grid.info.origin.position.y = -R
        grid.info.origin.position.z = 0.0

        data, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=np.linspace(-R, R, N + 1))
        data = np.where(data > 0, data * 4 + 20, 0)
        data = np.minimum(data, 100)
        grid.data = data.T.flatten().tolist()

        return grid

    def __call__(self, msg):
        r, th = self.msg2polar(msg)
        r, th = self.filter_r(r, th)

        p = self.polar2cartesian(r, th)
        p = self.filter_alpha(p)

        grid = self.make_grid(p)
        self.map_publisher.publish(grid)

        self.rate.sleep()


subscriber = rospy.Subscriber('/base_scan', LaserScan, LaserFilter())

rospy.spin()