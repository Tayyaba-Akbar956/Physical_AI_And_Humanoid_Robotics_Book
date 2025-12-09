---
sidebar_position: 1
title: ROS 2 Overview
---

# ROS 2 Overview

This chapter introduces ROS 2 (Robot Operating System 2), the middleware framework that enables communication between different software components in robotic systems. ROS 2 is the "nervous system" of modern robots, connecting perception, planning, and control components.

## Learning Objectives

- Understand what ROS 2 is and its role in robotic systems
- Identify the core concepts of ROS 2 architecture
- Recognize the advantages of using ROS 2 for robotics development
- Explain how ROS 2 differs from its predecessor ROS 1
- Prepare for hands-on experience with ROS 2 tools

## Introduction: The Robotic Nervous System

ROS 2 (Robot Operating System 2) is not an operating system in the traditional sense, but rather a flexible framework for developing robot software. It provides libraries, tools, and conventions that enable different software components to communicate and coordinate their behavior.

If we think of a robot as having a nervous system, ROS 2 serves as the neural pathways that allow different parts of the robot to communicate. Just as your brain communicates with your muscles, sensors, and other organs through your nervous system, ROS 2 allows the different software packages in a robot to exchange information.

### Why ROS 2 Matters

Modern robots are complex systems that require coordination between many different software components:
- Perception systems that process sensor data
- Planning systems that determine what actions to take
- Control systems that execute those actions
- User interfaces that allow human interaction
- Diagnostic systems that monitor robot health

Each of these components might be developed by different teams or even different organizations. ROS 2 provides a standard way for these components to communicate, making it possible to combine them into a functioning robot system.

### The Evolution from ROS 1 to ROS 2

ROS 2 is the successor to the original Robot Operating System (ROS 1). While ROS 1 established many of the core concepts still used today, ROS 2 was designed to address several key limitations:

- **Real-time support**: ROS 2 supports real-time applications critical for robot control
- **Deterministic behavior**: Improved timing guarantees for safety-critical applications
- **Multi-robot systems**: Better support for coordinating multiple robots
- **Security**: Built-in security features for deployment in real environments
- **Industry compatibility**: Better integration with industrial software practices

## Core Concepts

### Nodes

A **node** is a process that performs computation in a ROS 2 system. Each node typically performs a specific task:

- A camera driver node processes images from a camera
- A navigation node plans paths through an environment
- A motor control node sends commands to robot motors
- A perception node processes sensor data to detect objects

Nodes are the building blocks of ROS 2 applications, and complex robot behaviors emerge from the interaction of multiple nodes.

### Topics and Messages

**Topics** are named buses over which nodes exchange messages. Think of a topic as a specific channel of communication:

- `/camera/image_raw` - carries raw images from a camera
- `/cmd_vel` - carries velocity commands to a mobile base
- `/scan` - carries laser range measurements
- `/tf` - carries transformations between coordinate frames

**Messages** are the data structures that are sent over topics. Each topic has a specific message type that defines the structure of the data:

```python
# Example: geometry_msgs/Twist message for velocity commands
linear:
  x: 0.5     # Move forward at 0.5 m/s
  y: 0.0     # No sideways motion
  z: 0.0     # No vertical motion
angular:
  x: 0.0     # No rotation around x-axis
  y: 0.0     # No rotation around y-axis
  z: 0.2     # Rotate counterclockwise at 0.2 rad/s
```

### Publishers and Subscribers

Nodes communicate through a **publisher-subscriber** model:

- **Publishers** create and send messages to topics
- **Subscribers** receive messages from topics

This design allows nodes to operate independently - a publisher doesn't need to know how many subscribers exist, and subscribers don't need to know the source of their messages.

### Services

In addition to the asynchronous topic communication, ROS 2 supports **services** for synchronous request-response communication:

- A node sends a **request** to a service
- The service processes the request and sends back a **response**
- This is useful for operations that require confirmation or return specific results

### Actions

For long-running operations, ROS 2 provides **actions**:
- Like services, but for operations that take significant time
- Provide feedback during execution
- Support goal preemption (canceling long-running goals)

## Practical Implementation

### Example: Turtle Robot System

Let's examine a simple example of how ROS 2 components work together using the classic "turtle" tutorial:

```python
# turtle_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

class TurtleController(Node):
    def __init__(self):
        super().__init__('turtle_controller')
        
        # Publisher to send velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Subscriber to receive turtle pose
        self.pose_sub = self.create_subscription(
            Pose, '/turtle1/pose', self.pose_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.target_x = 5.0  # Target x position
        self.kp = 1.0        # Proportional control gain
        
    def pose_callback(self, msg):
        # Store current pose
        self.current_pose = msg
        
    def control_loop(self):
        # Simple proportional controller
        msg = Twist()
        msg.linear.x = self.kp * (self.target_x - self.current_pose.x)
        self.cmd_vel_pub.publish(msg)

def main():
    rclpy.init()
    controller = TurtleController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

In this example:
- The node publishes velocity commands to `/turtle1/cmd_vel`
- It subscribes to pose information from `/turtle1/pose`
- It runs a control loop every 0.1 seconds to adjust the turtle's position

### Basic ROS 2 Commands

Once you have ROS 2 installed, you'll frequently use these commands:

- `ros2 run <package> <executable>` - Run a node
- `ros2 topic list` - List active topics
- `ros2 topic echo <topic_name>` - Print messages from a topic
- `ros2 node list` - List active nodes
- `ros2 launch <package> <launch_file>` - Launch multiple nodes at once

## Hands-on Exercise

1. **Conceptual Exercise**: Draw a simple robot system (e.g., a mobile robot with camera, laser scanner, and wheels) and identify what ROS 2 nodes, topics, and messages would be needed for basic operation.

2. **Research Exercise**: Look up one real-world robot that uses ROS/ROS 2 and identify the types of nodes and topics it likely uses.

3. **Comparison Exercise**: Compare ROS 2 to another communication framework you're familiar with (e.g., REST APIs, message queues) in terms of:
   - Real-time capabilities
   - Network transparency
   - Type safety
   - Ease of use

## Key Takeaways

- ROS 2 provides the communication infrastructure for robotic systems
- Nodes are the basic computational units that communicate through topics
- Topics use a publisher-subscriber model for asynchronous communication
- Services and actions provide synchronous and long-running communication
- ROS 2 enables modular robot software design
- Understanding ROS 2 is essential for modern robotics development

## Further Reading

- ROS 2 Documentation: https://docs.ros.org/
- "Programming Robots with ROS" by Morgan Quigley, Brian Gerkey, and William Smart
- "Effective Robotics Programming with ROS" by Anil Mahtani, Enrique Fernandez, and Luis Sanchez

## Next Steps

Continue to Chapter 2: Nodes, Topics, and Services to dive deeper into the communication mechanisms that make ROS 2 powerful.