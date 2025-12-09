---
sidebar_position: 2
title: Nodes, Topics, and Services
---

# Nodes, Topics, and Services

This chapter dives deep into the three fundamental communication mechanisms in ROS 2: nodes (computational units), topics (asynchronous communication), and services (synchronous communication). Understanding these concepts is essential for building distributed robotic systems.

## Learning Objectives

- Explain the role and implementation of nodes in ROS 2
- Implement publishers and subscribers for topic communication
- Design and implement services for synchronous communication
- Analyze the trade-offs between topics and services
- Apply appropriate communication patterns to robot design problems

## Introduction: Building Blocks of Communication

ROS 2's communication system is built on three primary patterns that handle different types of information exchange:

- **Nodes** provide the computational framework
- **Topics** enable asynchronous, decoupled data flow
- **Services** support synchronous request-response interactions

These patterns reflect different requirements in robotic systems:
- Sensor data flows continuously and asynchronously (topics)
- Control commands are often immediate and require confirmation (services)
- Different computational tasks run independently but need to share information (nodes)

## Core Concepts

### Nodes: The Computational Foundation

A ROS 2 node is an instance of a computational process that performs specific tasks in a robot system. Nodes are the containers for your robot's functionality:

- **Encapsulation**: Each node encapsulates specific functionality (navigation, perception, etc.)
- **Identity**: Nodes have unique names within the ROS 2 graph
- **Resource management**: Nodes manage their own timers, callbacks, and execution contexts
- **Communication interface**: Nodes contain publishers, subscribers, services, and clients

Each node operates independently but coordinates with other nodes through the ROS 2 communication infrastructure.

### Topics: Asynchronous Data Streams

Topics implement a publish-subscribe communication pattern where:

**Publishers**:
- Send messages to named topics
- Don't know or care if there are subscribers
- Send data at rates appropriate for the application

**Subscribers**:
- Receive messages from named topics
- Don't know the source of messages
- Process messages as they arrive

**Characteristics of Topic Communication**:
- **Loose coupling**: Publishers and subscribers don't need to run simultaneously
- **Broadcast**: One publisher can serve multiple subscribers
- **Real-time friendly**: Minimal overhead for continuous data streams
- **One-way**: Data flows from publisher to subscriber only

### Quality of Service (QoS)

ROS 2 topics support Quality of Service settings that control communication behavior:

- **Reliability**: Reliably (all messages delivered) vs. best-effort (some messages may be lost)
- **Durability**: Transient-local (messages stored for late joiners) vs. volatile (no storage)
- **History**: Keep-all vs. keep-last N messages
- **Rate**: Maximum rate for throttling messages

### Services: Request-Response Communication

Services implement synchronous request-response communication:

- A **service client** sends a request and waits for a response
- A **service server** receives the request, processes it, and sends a response
- Communication is point-to-point (one client to one server)

**Use cases for services**:
- Configuration changes that require confirmation
- Computations that must complete before proceeding
- Discrete actions (e.g., "take snapshot", "calibrate sensor")
- Operations where success/failure matters for subsequent actions

### Actions: Long-running Operations

Actions are like services but for operations that take significant time:

- Clients can send goals and receive feedback during execution
- Clients can cancel goals before completion
- Useful for navigation tasks, manipulation sequences, or calibration procedures

## Practical Implementation

### Creating a Node

Here's a complete example of a ROS 2 node that publishes sensor data:

```python
# sensor_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('laser_sensor_publisher')
        
        # Create publisher
        self.publisher = self.create_publisher(
            LaserScan, 
            '/scan',  # Topic name
            10        # Queue size
        )
        
        # Create timer for periodic publishing
        self.timer = self.create_timer(0.1, self.publish_scan)  # 10 Hz
        
        # Sensor parameters
        self.angle_min = -1.57  # -90 degrees
        self.angle_max = 1.57   # 90 degrees
        self.angle_increment = 0.017  # 1 degree
        self.scan_time = 0.1
        self.time_increment = 0.0
        
    def publish_scan(self):
        msg = LaserScan()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_frame'
        
        # Set sensor parameters
        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.angle_increment = self.angle_increment
        msg.scan_time = self.scan_time
        msg.time_increment = self.time_increment
        msg.range_min = 0.1
        msg.range_max = 10.0
        
        # Generate simulated scan data
        num_readings = int((self.angle_max - self.angle_min) / self.angle_increment) + 1
        msg.ranges = []
        for i in range(num_readings):
            # Simulate distance with some noise
            distance = 2.0 + random.uniform(-0.1, 0.1)
            msg.ranges.append(distance)
        
        # Publish the message
        self.publisher.publish(msg)
        self.get_logger().info(f'Published laser scan with {len(msg.ranges)} readings')

def main():
    rclpy.init()
    node = SensorPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber

A subscriber node that processes the published laser scan:

```python
# obstacle_detector.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        
        # Create subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.subscription  # Prevent unused variable warning
        
    def scan_callback(self, msg):
        # Check if there are obstacles within 1 meter
        safe_distance = 1.0
        obstacles_detected = False
        
        for i, range_val in enumerate(msg.ranges):
            if not math.isnan(range_val) and range_val < safe_distance:
                angle = msg.angle_min + i * msg.angle_increment
                self.get_logger().warn(
                    f'Obstacle detected at {range_val:.2f}m, angle {math.degrees(angle):.1f}°'
                )
                obstacles_detected = True
        
        if not obstacles_detected:
            self.get_logger().info('Path clear')

def main():
    rclpy.init()
    node = ObstacleDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Service

A service for simple robot commands:

```python
# robot_service.py
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from example_interfaces.srv import SetBool

class RobotService(Node):
    def __init__(self):
        super().__init__('robot_service')
        
        # Create service
        self.srv = self.create_service(
            SetBool, 
            'robot_enable', 
            self.enable_callback
        )
        self.enabled = False
        
    def enable_callback(self, request, response):
        self.enabled = request.data
        if self.enabled:
            self.get_logger().info('Robot enabled')
        else:
            self.get_logger().info('Robot disabled')
            
        response.success = True
        response.message = f'Robot {"enabled" if self.enabled else "disabled"}'
        return response

def main():
    rclpy.init()
    node = RobotService()
    
    # Use multithreaded executor since services run on separate threads
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Communication Pattern Selection

Choose the appropriate communication pattern based on your needs:

**Use Topics for**:
- Sensor data streams
- Continuous state updates
- Broadcasting information to multiple consumers
- When real-time performance is critical
- When the publisher doesn't need to know about subscribers

**Use Services for**:
- Discrete actions with clear outcomes
- Configuration changes requiring confirmation
- Computations that must complete successfully
- When synchronous response is needed
- When there's a clear one-to-one relationship between client and server

**Use Actions for**:
- Long-running operations (navigation, manipulation)
- When you need feedback during execution
- When operations can be canceled
- For goal-oriented behaviors

## Hands-on Exercise

1. **Design Exercise**: Design a simple robot system (e.g., a mobile robot that navigates to goal positions) and specify the nodes, topics, and services required. For each communication connection, justify why you chose the pattern you did.

2. **Implementation Exercise**: Build a ROS 2 node that publishes messages to a topic, and another node that subscribes to that topic and performs an action based on the received data.

3. **Analysis Exercise**: For a robot performing a sequence of tasks (e.g., navigate to location A, pick up object, navigate to location B, place object), identify which steps should use topics, services, and actions, and explain your reasoning.

## Key Takeaways

- Nodes encapsulate computational functionality in ROS 2
- Topics provide asynchronous, decoupled communication
- Services provide synchronous request-response communication
- Quality of Service settings allow fine-tuning of communication behavior
- Appropriate choice of communication pattern is critical for system performance
- Topics are for continuous data streams, services for discrete actions

## Further Reading

- "ROS Robotics Projects" by Anis Koubaa
- ROS 2 Design documentation on Quality of Service policies
- "Programming Robots with ROS" by Quigley et al.

## Next Steps

Continue to Chapter 3: Python and rclpy to learn how to implement ROS 2 nodes using Python.