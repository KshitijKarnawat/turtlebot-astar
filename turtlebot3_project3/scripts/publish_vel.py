#!/usr/bin/env python3

import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pandas as pd


class MyNode(Node):
    def __init__(self):
        super().__init__('astar_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.idx = 0
        self.wheel_r = 0.033
        self.wheel_base = 0.16

        
    

def main():
  rclpy.init()
  node = MyNode()

  # Spin in a separate thread
  thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
  thread.start()

  rate = node.create_rate(0.45)

  rpm = pd.read_csv("/home/nova/ros2_ws/src/turtlebot3_project3/scripts/rpm.csv", delimiter=',', header=None, dtype=float)

  left_rpm = rpm.iloc[node.idx][0]
  right_rpm = rpm.iloc[node.idx][1]
  
  robot_vel = Twist()

  lin_vel = (node.wheel_r*(left_rpm + right_rpm))/2.0
  ang_vel = (right_rpm - left_rpm)*node.wheel_r/node.wheel_base

  try:
      while rclpy.ok() and node.idx < len(rpm):
          print(left_rpm, right_rpm)
          robot_vel.linear.x = lin_vel
          robot_vel.angular.z = ang_vel
          node.publisher_.publish(robot_vel)
          rate.sleep()
          node.idx += 1
          left_rpm = rpm.iloc[node.idx][0] * 0.45
          right_rpm = rpm.iloc[node.idx][1] * 0.45 
          lin_vel = (node.wheel_r*(left_rpm + right_rpm))/2.0
          ang_vel = (right_rpm - left_rpm)*node.wheel_r/node.wheel_base

  except KeyboardInterrupt:
      robot_vel.linear.x = 0.0
      robot_vel.angular.z = 0.0
      node.publisher_.publish(robot_vel)
      pass

  rclpy.shutdown()
  thread.join()

if __name__ == '__main__':
    main()