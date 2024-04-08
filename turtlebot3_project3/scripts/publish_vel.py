#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pandas, math

class AStarCommander(Node):

  def __init__(self):
    super().__init__('astar_commander')
    self.rpms = pandas.read_csv("../../part1/rpm.csv", delimiter=',', header=None, dtype=float)
    self.index = 0
    self.path_published = False
    self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
    self.publish_timer = self.create_timer(1, self.publish_velocities)

  def publish_velocities(self):
    velocity = Twist()

    velocity.linear.x = 0.0
    velocity.linear.y = 0.0
    velocity.linear.z = 0.0

    velocity.angular.x = 0.0
    velocity.angular.y = 0.0
    velocity.angular.z = 0.0

    if (self.index == len(self.rpms)):
      self.cmd_vel_pub.publish(velocity)
      self.path_published = True
      print("Finished publishing velocities")
      return

    left_rpm = float(self.rpms.iloc[[self.index]][0])
    right_rpm = float(self.rpms.iloc[[self.index]][1])

    velocity.angular.z = math.radians((0.038 / 0.354) * (right_rpm - left_rpm))
    velocity.linear.x = (0.038 / 2) * (left_rpm + right_rpm) * math.cos(velocity.angular.z)
    velocity.linear.y = (0.038 / 2) * (left_rpm + right_rpm) * math.sin(velocity.angular.z)

    self.cmd_vel_pub.publish(velocity)
    self.index += 1

def main(args=None):
  rclpy.init(args=args)

  astar_commander_node = AStarCommander()
  print("Publishing velocities from CSV...")
  rclpy.spin(astar_commander_node)

  while (astar_commander_node.path_published == False): pass

  astar_commander_node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()
