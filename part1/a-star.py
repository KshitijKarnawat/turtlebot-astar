"""
a_star_kshitij_abhishek.py

@breif:     This module implements A-star algorithm for finding the shortest path in a graph.
@author:    Kshitij Karnawat, Abhishekh Reddy
@date:      7th April 2024
@version:   3.0

@github:    https://github.com/KshitijKarnawat/a-star-path-planner
"""

import os
import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import csv
import time


class NewNode:
    """Class to represent a node in the graph
    """
    def __init__(self, pose, parent, cost_to_go, cost_to_come, left_wheel_rpm, right_wheel_rpm, path):
        """Initializes the node with its coordinates, parent and cost

        Args:
            coord (tuple): Coordinates of the node along with the angle
            parent (NewNode): Parent node of the current node
            cost_to_go (float): Cost to reach the current node
            cost_to_come (float): A-Star Hueristic for the current node (Eucledian Distance)
        """
        self.pose = pose
        self.parent = parent
        self.cost_to_go = cost_to_go
        self.cost_to_come = cost_to_come
        self.total_cost = cost_to_come + cost_to_go
        self.left_wheel_rpm = left_wheel_rpm
        self.right_wheel_rpm = right_wheel_rpm
        self.path = path

def in_obstacles(pose, clearance):
    """Checks if the given coordinates are in obstacles

    Args:
        coord (tuple): Coordinates to check

    Returns:
        bool: True if the coordinates are in obstacles, False otherwise
    """
    # Set Max and Min values for x and y
    x_max, y_max = 600, 200
    x_min, y_min = 0, 0

    x, y, heading = pose

    bloat = clearance

    if (x < x_min + bloat) or (x > x_max - bloat) or (y < y_min + bloat) or (y > y_max - bloat):
        # print("Out of bounds")
        return True

    # Rectangle 1
    elif (x >= 150 - bloat and x <= 175 + bloat) and (y >= 100 - bloat and y <= 200):
        # print("In Obstacle 1")
        return True

    # Rectangle 2
    elif (x >= 250 - bloat and x <= 275 + bloat) and (y >= 0 and y <= 100 + bloat):
        # print("In Obstacle 2")
        return True

    # Circle
    elif (x - 420) ** 2 + (y - 120) ** 2 <= (60 + bloat) ** 2:
        # print("In Obstacle 3")
        return True    

    return False

def near_goal(start, goal, threshold):
    """Checks if the start node is near the goal node

    Args:
        start (tuple): Start coordinates
        goal (tuple): Goal coordinates
        threshold (int): Threshold distance

    Returns:
        bool: True if the start node is near the goal node, False otherwise
    """
    return calc_euclidian_distance(start, goal) <= threshold

def calc_euclidian_distance(start, goal):
    """Calculates the euclidian distance between two points

    Args:
        start (tuple): Start coordinates
        goal (tuple): Goal coordinates

    Returns:
        float: Euclidian distance between the two points
    """
    return np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)

def get_child_nodes(node, goal_pose, clearance, w1, w2):
    """Generates all possible child nodes for the given node

    Args:
        node (NewNode): Node to generate child nodes from
        goal_coord (tuple): Coordinates of the goal node

    Returns:
        list: List of child nodes and their costs
    """

    # child nodes list
    child_nodes = []
    actions = [[0, w1], [w1, 0], [w1, w1], [0, w2], [w2, 0], [w2, w2],
               [w1, w2], [w2, w1]]

    # Create all possible child nodes
    for action in actions:
        left_wheel_rpm, right_wheel_rpm = action
        left_wheel_rpm = left_wheel_rpm * (np.pi / 30)
        right_wheel_rpm = right_wheel_rpm * (np.pi / 30)
        child_path = []
        
        # Constants defined for the robot
        t = 0
        r = 3.3     # radius of the wheel
        L = 16      # distance between wheels
        dt = 0.1
        
        valid = False
        x, y, theta = node.pose
        theta_rad = np.deg2rad(theta)
        action_cost = 0
        Xn = x
        Yn = y

        while t < 1:
            t = t + dt
            theta_rad += (r / L) * (right_wheel_rpm - left_wheel_rpm) * dt
            Xs, Ys = Xn, Yn
            # Differential Drive Constraints
            Xn = Xs + 0.5 * r * (left_wheel_rpm + right_wheel_rpm) * np.cos(theta_rad) * dt
            Yn = Ys + 0.5 * r * (left_wheel_rpm + right_wheel_rpm) * np.sin(theta_rad) * dt
            child_path.append([[Xs, Xn], [Ys, Yn]])
            if not in_obstacles((Xn, Yn, theta_rad), clearance):
                valid = True
                break
        
        if valid:
            theta = np.rad2deg(theta_rad)
            action_cost = calc_euclidian_distance((Xn, Yn), (Xs, Ys))
            cost_to_go = calc_euclidian_distance((Xn, Yn), goal_pose)
            
            child_node = NewNode((int(round(Xn, 0)), int(round(Yn, 0)), theta),
                                  node,
                                  cost_to_go,
                                  node.cost_to_come + action_cost, 
                                  left_wheel_rpm, 
                                  right_wheel_rpm, 
                                  child_path
                                  )
            
            child_nodes.append((child_node, action_cost))

    return child_nodes

def astar(start_pose, goal_pose, clearance, threshold, w1, w2):
    """Finds the shortest path from start to goal using Dijkstra's algorithm

    Args:
        start (tuple): Start coordinates
        goal (tuple): Goal coordinates

    Returns:
        list: A list of explored nodes
        list: A list of coordinates representing the shortest path
    """
    # Initialize open and closed lists
    open_list = []
    open_list_info = {}
    closed_list = []
    closed_list_info = {}
    path = []
    explored_nodes = []

    # Create start node and add it to open list
    start_node = NewNode(start_pose, 
                         None, 
                         calc_euclidian_distance(start_pose, goal_pose), 
                         0,
                         0, 
                         0, 
                         []
                         )
    
    open_list.append((start_node, start_node.total_cost))
    open_list_info[start_node.pose] = start_node
    start_time = time.time()
    while open_list:
        # Get the node with the minimum total cost and add to closed list
        open_list.sort(key=lambda x: x[1]) # sort open list based on total cost
        current_node, _ = open_list.pop(0)
        open_list_info.pop(current_node.pose)
        closed_list.append(current_node)
        closed_list_info[current_node.pose] = current_node

        # Check if goal reached
        if near_goal(current_node.pose, goal_pose, threshold):
            end_time = time.time()
            print("Time taken by A-Star:", end_time - start_time)
            path, rpm, trajectory = backtrack_path(current_node)
            return explored_nodes, path, rpm, trajectory

        else:
            children = get_child_nodes(current_node, goal_pose, clearance, w1, w2)
            for child, child_cost in children:
                if child.pose in closed_list_info:
                    del child
                    continue

                if child.pose in open_list_info:
                    if child_cost + current_node.cost_to_come < open_list_info[child.pose].cost_to_come:
                        open_list_info[child.pose].cost_to_come = child_cost + current_node.cost_to_come
                        open_list_info[child.pose].total_cost = open_list_info[child.pose].cost_to_come + open_list_info[child.pose].cost_to_go
                        open_list_info[child.pose].parent = current_node
                else:
                    child.parent = current_node
                    open_list.append((child, child.total_cost))
                    open_list_info[child.pose] = child

                    explored_nodes.append(child)

    end_time = time.time()
    print("Time taken by A-Star:", end_time - start_time)
    return explored_nodes, None, None, None

# Reused from Previous Assignment
def backtrack_path(goal_node):
    """Backtracking algorithm for Dijkstra's algorithm

    Args:
        goal_node (NewNode): Goal node

    Returns:
        list: A list of coordinates representing the shortest path
    """
    path = []
    rpm = []
    trajectory = []
    parent = goal_node
    while parent!= None:
        path.append((parent.pose[0], parent.pose[1]))
        rpm.append((parent.left_wheel_rpm, parent.right_wheel_rpm))
        parent = parent.parent
    return path[::-1], rpm[::-1], trajectory[::-1]

# Reused from Previous Assignment
def vizualize(start, goal, path, explored_nodes, rpm, trajectory, animate=False):
    """Vizualizes the path and explored nodes

    Args:
        game_map (numpy array): A 2D array representing the game map
        start (tuple): Start coordinates
        goal (tuple): Goal coordinates
        path (list): A list of coordinates representing the shortest path
        explored_nodes (list): A list of explored nodes
    """
    count = 0
    plt.rcParams["figure.figsize"] = [30,20]
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.margins(0)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 200)
    if animate:
        if not os.path.exists('output'):
            os.makedirs('output')
        save = cv.VideoWriter('turtlebot.avi',cv.VideoWriter_fourcc('M','J','P','G'),10,(2160,1440))
    
    # Plot start and goal points
    ax.scatter(start[0], start[1], color='red', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='green', s=100, label='Goal')

    # Plot obstacles
    x, y = np.meshgrid(np.arange(0, 600), np.arange(0, 200))
    rect1 = plt.Rectangle((150, 100), 25, 100, color='black')
    rect2 = plt.Rectangle((250, 0), 25, 100, color='black')
    circle = plt.Circle((420, 120), 60, color='black')
    ax.add_artist(rect1)
    ax.add_artist(rect2)
    ax.add_artist(circle)

    # Plot boundaries
    bound1 = (x<=5)
    bound2 = (x>=595)
    bound3 = (y<=5)
    bound4 = (y>=195)
    ax.fill(x[bound1], y[bound1], color='black')
    ax.fill(x[bound2], y[bound2], color='black')
    ax.fill(x[bound3], y[bound3], color='black')
    ax.fill(x[bound4], y[bound4], color='black')
    ax.set_aspect(1)

    if animate:
        plt.savefig('output/output_image_' + str(count) + '.png')
        count += 1

    start_time = time.time()

    # Plot explored nodes
    for node in explored_nodes:
        for x, y in node.path:
            ax.scatter(x, y, color='blue')
            if animate:
                if count % 10 == 0:
                    plt.savefig('output/output_image_' + str(count) + '.png')
                count += 1

    # Plot path
    for point, bt in zip(trajectory, path):
        ax.scatter(bt[0], bt[1], color='yellow')
        for xi, yi in point.path:
            ax.scatter(xi, yi, color='red')
        if animate:
            plt.savefig('output/output_image_' + str(count) + '.png')
            count += 1

    if animate:
        for filename in os.listdir('output'):
            img = cv.imread(os.path.join("output",filename))
            save.write(img)
        save.release()
    end_time = time.time()
    print("Time taken to visualize:", end_time - start_time)


def write_rpm_to_file(rpm):
    with open('rpm.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for w1,w2 in rpm:    
            writer.writerow([round(w1,3),round(w2,3)])

def main():

    parser = argparse.ArgumentParser(description='Animate on/off')
    parser.add_argument('--animate', type=bool , default=False, help="Create the animation video default is False")
    args = parser.parse_args()

    # get start and end points from user
    start_point = (int(input("Enter x coordinate of start point: ")), int(input("Enter y coordinate of start point: ")), int(input("Enter the start angle of the robot in multiples of 30deg(0 <= theta <= 360): ")))
    goal_point = (int(input("Enter x coordinate of goal point: ")), int(input("Enter y coordinate of goal point: ")), int(input("Enter the goal angle of the robot in multiples of 30deg(0 <= theta <= 360): ")))
    clearance = int(input("Enter the clearance for robot: "))
    threshold = int(input("Enter the threshold distance for goal: "))
    w1 = int(input("Enter w1: "))
    w2 = int(input("Enter w2: "))

    # Check if start and goal points are in obstacles
    if in_obstacles(start_point, clearance):
        print("Start point is in obstacle")
        return

    if in_obstacles(goal_point, clearance):
        print("Goal point is in obstacle")
        return

    # find shortest path
    explored_nodes, shortest_path, rpm, trajectory = astar(start_point, goal_point, clearance, threshold, w1, w2)
    if shortest_path == None:
        print("No path found")

    # write rpm to file
    if rpm != None:
        write_rpm_to_file(rpm)

    # visualize path
    if shortest_path != None:
        vizualize(start_point, goal_point, shortest_path, explored_nodes, rpm, trajectory, args.animate)

if __name__ == "__main__":
    main()