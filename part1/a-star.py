"""
a_star_kshitij_abhishek.py

@breif:     This module implements A-star algorithm for finding the shortest path in a graph.
@author:    Kshitij Karnawat, Abhishekh Reddy
@date:      7th April 2024
@version:   3.0

@github:    https://github.com/KshitijKarnawat/a-star-path-planner
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math
import csv
import os

class Node:
    """Class to represent a node in the search tree

    Args:
        pose (tuple): x, y, theta of the node
        parent (Node): parent node
        cost_to_come (float): cost to come to the node
        total_cost (float): total cost of the node
        left_wheel (float): left wheel speed
        right_wheel (float): right wheel speed
        path (list): path taken by the robot to reach the node
    """
    def __init__(self, pose, parent, cost_to_come, total_cost, left_wheel, right_wheel, path):
        self.pose = pose
        self.parent = parent
        self.cost_to_come = cost_to_come
        self.total_cost = total_cost
        self.left_wheel = left_wheel
        self.right_wheel = right_wheel
        self.path = path

def calc_euclidean_dist(start, goal):
    """Calculate euclidean distance between two points

    Args:
        start (tuple): Start point (x, y)
        goal (tuple): Goal point (x, y)

    Returns:
        float: Euclidean distance between start and goal
    """
    return np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)

def in_obstacle(pose, clearence):
    """Check if the robot is in the obstacle space

    Args:
        pose (tuple): Pose of the robot (x, y, theta)
        clearence (int): Clearence of the robot

    Returns:
        bool: True if robot is in obstacle space, False otherwise
    """

    x_max, y_max = [600 + 1, 200 + 1]
    x_min, y_min = [0, 0]
    x, y, th = pose
    
    # walls
    if x <= clearence or x >= x_max - clearence or y <= clearence or y >= y_max - clearence:
        return False
    
    # left rectangle
    elif x>= 150 - clearence and x<= 175 + clearence and y >= 100 - clearence and y <= y_max - clearence:
        return False
    
    # right rectangle
    elif x >= 250 - clearence and x<= 275 + clearence and y >= y_min + clearence  and y <= 100 + clearence:
        return False
    
    # circle
    elif (x-420)**2 + (y-120)**2 <=(60+clearence)**2:
        return False
    
    else:
        return True
    
def get_child(node, goal_pose, clearence, w1, w2):
    """Get child nodes for the current node

    Args:
        node (Node): Current node
        goal_pose (tuple): Goal pose (x, y, theta)
        clearence (int): Clearence of the robot
        w1 (int): Left wheel speed
        w2 (int): Right wheel speed

    Returns:
        list: List of child nodes
    """

    child_list = []
    actionSet = [[0,w1], [w1,0], [w1,w1], [0,w2],
                 [w2,0], [w2,w2], [w1,w2], [w2,w1]]
    
    
    for action in actionSet:
        left_wheel,right_wheel = action 
        left_wheel= left_wheel*(math.pi/30)
        right_wheel= right_wheel*(math.pi/30)
        child_path = []

        # Robot parameters
        t = 0
        r = 3.3     # radius of wheel
        L = 16      # distance between wheels
        dt = 0.1
        valid = True

        x, y, theta = node.pose
        theta_rad = np.deg2rad(theta)
        child_cost = 0
        Xn = x
        Yn = y

        while t<1:
            t = t + dt
            Xs,Ys = Xn,Yn
            theta_rad += (r / L) * (right_wheel - left_wheel) * dt
            Xn = Xs + 0.5* r * (left_wheel + right_wheel) * math.cos(theta_rad) * dt
            Yn =  Ys + 0.5 * r * (left_wheel + right_wheel) * math.sin(theta_rad) * dt
            child_path.append([[Xs, Xn], [Ys, Yn]])

            if not in_obstacle((Xn, Yn, theta_rad), clearence):
                valid = False
                break

        if valid:
            theta = np.rad2deg(theta_rad)
            child_cost = calc_euclidean_dist((Xs, Ys), (Xn, Yn))
            cost_to_go = np.linalg.norm(np.asarray((Xn, Yn)) - np.asarray((goal_pose[0], goal_pose[1])))
            child = Node((int(round(Xn, 0)), int(round(Yn, 0)), theta), 
                        node, 
                        node.cost_to_come + child_cost,
                        node.cost_to_come + child_cost + cost_to_go, 
                        left_wheel, 
                        right_wheel, 
                        child_path
                        )
            
            child_list.append((child_cost, cost_to_go, child))

    return child_list

def backtrack(current):
    path = []
    speeds= []
    trajectory = []
    parent = current
    while parent != None:
        path.append(parent.pose)
        speeds.append((parent.left_wheel,parent.right_wheel))
        trajectory.append(parent)
        parent = parent.parent
    return path, speeds, trajectory

def astar(start, goal, clearence , w1, w2, threshold):
    """Implement A* algorithm to find the path from start to goal

    Args:
        start (Node): Start node
        goal (Node): Goal node
        clearence (int): Clearence of the robot
        w1 (int): Speed of wheel for action
        w2 (int): Speed of wheel for action
        threshold (int): Threshold to reach goal

    Returns:
        list: list of nodes in the path
        list: list of speeds of left and right wheels
        list: list of nodes in the trajectory
        list: list of explored nodes
    """

    open_list = []
    open_list_info = {}
    closed_list = []
    closed_list_info = {}
    explored_nodes = []

    initial_node = Node(start, 
                        None, 
                        0, 
                        calc_euclidean_dist(start, goal), 
                        0,
                        0, 
                        []
                        )
    
    open_list.append((initial_node.total_cost, initial_node))
    open_list_info[initial_node.pose] = initial_node

    start = time.time()
    while open_list:
        open_list.sort(key=lambda x: x[0])
        _, current = open_list.pop(0)
        open_list_info.pop(current.pose)
        closed_list.append(current)
        closed_list_info[current.pose] = current

        if calc_euclidean_dist(current.pose, goal) <= threshold:
            pathTaken, rpms, trajectory = backtrack(current)
            end = time.time()
            print("Path Found")
            print('Time taken to execute algorithm: ',(end - start)," sec")
            return (pathTaken,rpms,trajectory, explored_nodes)
        
        else:
            childList = get_child(current, goal, clearence, w1, w2)
            for child_cost, cost_to_go, child in childList:
                if child.pose in closed_list_info:
                    del child
                    continue

                if child.pose in open_list_info:
                    if open_list_info[child.pose].cost_to_come > current.cost_to_come + child_cost:
                        open_list_info[child.pose].parent = current
                        open_list_info[child.pose].cost_to_come = current.cost_to_come + child_cost
                        open_list_info[child.pose].total_cost = open_list_info[child.pose].cost_to_come + cost_to_go
                
                else:
                    open_list.append((child.total_cost, child))
                    open_list_info[child.pose] = child
                    explored_nodes.append(child)

    end = time.time()
    print("Time taken to execute algorithm: ",(end - start)," sec")
    return None, None, None, None

def visualization(explored_nodes, shortest_path, trajectory, start_pose, goal_pose, threshold, animate=False):
    """Visualize the path taken by the robot

    Args:
        explored_nodes (List): Explored nodes
        pathTaken (List): List of nodes in the path
        trajectory (List): List of nodes in the trajectory
        start_pose (tuple): Start pose of the robot
        goal_pose (tuple): Goal pose of the robot
        animate (bool, optional): Creates a video if True. Defaults to False.
    """

    counter = 0
    plt.rcParams["figure.figsize"] = [30,20]
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.margins(0)
    plt.xlim(0,600)
    plt.ylim(0,200)

    if animate:
        if not os.path.exists('animate'):
            os.makedirs('animate')
        save = cv2.VideoWriter('turtlebot_k.avi', 
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               10,
                               (1280,720)
                               )
    
    # set goal and start
    ax.scatter(start_pose[0],start_pose[1],color = "red")
    goal = plt.Circle((goal_pose[0],goal_pose[1]), threshold,color = "green")
    ax.add_artist(goal)

    # draw obstacle space
    x, y = np.meshgrid(np.arange(0, 600), np.arange(0, 200))

    rectangle1 = plt.Rectangle((150, 100), 25, 100, fc='black')
    rectangle2 = plt.Rectangle((250, 0), 25, 100, fc='black')    
    cc = plt.Circle(( 420 , 120 ), 60, color = "black") 

    ax.add_artist(rectangle1)
    ax.add_artist(rectangle2)
    ax.add_artist( cc )

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
        plt.savefig("animate/animateImg"+str(counter)+".png")
        counter += 1

    start_time = time.time()
    # to visualise child exploration
    for node in explored_nodes:
        for xe,ye in node.path:
            ax.plot(xe,ye, color="blue")
        if animate:
            plt.savefig("animate/animateImg"+str(counter)+".png")
            counter += 1
            
    # to visualize backtrack path
    for point,backtrack_path in zip(trajectory ,shortest_path):
        ax.scatter(backtrack_path[0],backtrack_path[1], color="black")
        for xt,yt in point.path:
            ax.plot(xt,yt, color="green")
        if animate:
            plt.savefig("animate/animateImg"+str(counter)+".png")
            counter += 1
    
    if animate:
        for filename in os.listdir("animate"):
            path = os.path.join("animate",filename)
            img = cv2.imread(path)
            img = cv2.resize(img,(1280,720))
            save.write(img)
        save.release()

    end_time = time.time()
    print('Time taken to visualize: ',(end_time - start_time)," sec")
    plt.show()

def export_rpms(rpms):
    """Export the rpms to a csv file

    Args:
        rpms (List): List of rpms
    """

    with open('../turtlebot3_project3/scripts/rpm.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for w1,w2 in rpms[::-1]:    
            writer.writerow([round(w1,3),round(w2,3)])
        writer.writerow([0,0])
        writer.writerow([0,0])

def main():

    start_pose= (int(input('Enter start x-coordinate: ')), int(input('Enter start y-coordinate: ')),0)
    goal_pose= (int(input('Enter goal x-coordinate: ')), int(input('Enter goal y-coordinate: ')),0)
    clearence = int(input('Enter clearance (robot radius + bloat): '))
    threshold = int(input('Enter goal threshold: '))
    w1 = int(input('Enter w1: '))
    w2 = int(input('Enter w2: '))
    
    if not in_obstacle(start_pose, clearence):
        print("Start pose is in obstacle space")
        return
    
    if not in_obstacle(goal_pose, clearence):
        print("Goal pose is in obstacle space")
        return

    pathTaken, rpms, trajectory, explored_nodes = astar(start_pose, goal_pose, clearence, w1, w2, threshold)
    if pathTaken == None:
        print("Path not found exiting...")
        return
    
    if rpms != None:
        export_rpms(rpms)

    if pathTaken != None:
        animate = False
        print("Visualisation: ", animate)
        visualization(explored_nodes, pathTaken, trajectory, start_pose, goal_pose, threshold, animate)

if __name__ == '__main__':
    main()