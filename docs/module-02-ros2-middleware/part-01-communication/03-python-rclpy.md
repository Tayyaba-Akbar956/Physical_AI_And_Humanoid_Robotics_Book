---
sidebar_position: 3
title: Python and rclpy
---

# Python and rclpy

This chapter focuses on implementing ROS 2 nodes using Python and the rclpy client library. Python is one of the most popular languages for robotics development, and rclpy provides the Python interface to the ROS 2 client library implementation.

## Learning Objectives

- Set up Python development environment for ROS 2
- Create nodes using rclpy with proper lifecycle management
- Implement publishers, subscribers, services, and clients
- Apply Python-specific patterns and best practices in ROS 2
- Debug and test Python ROS 2 nodes effectively

## Introduction: Python in Robotics

Python has become one of the dominant languages in robotics and AI development for several reasons:

- **Rapid prototyping**: Python's simplicity allows quick development and testing of robotic algorithms
- **Rich ecosystem**: Extensive libraries for computer vision, machine learning, and scientific computing
- **Community support**: Large robotics and AI communities with extensive resources
- **Integration**: Easy integration with other tools and systems

The `rclpy` package provides the Python client library for ROS 2, allowing Python developers to create ROS 2 nodes, publishers, subscribers, services, and clients.

## Core Concepts

### rclpy Architecture

The `rclpy` library provides a Python interface to the underlying ROS 2 client library implementation:

- **Nodes**: The basic execution unit in ROS 2
- **Publishers**: Send messages to topics
- **Subscribers**: Receive messages from topics
- **Services**: Provide synchronous request-response functionality
- **Clients**: Make synchronous requests to services
- **Actions**: Handle long-running goal-oriented tasks
- **Timers**: Execute callbacks at regular intervals
- **Parameters**: Store and manage node configuration

### Node Creation and Lifecycle

Every ROS 2 Python node inherits from the `rclpy.node.Node` class:

- The node constructor takes the node name and optional parameters
- Nodes manage their own resources and must be properly destroyed
- The rclpy execution model handles callback scheduling and message processing

### Threading and Concurrency

ROS 2 Python nodes can use different execution models:

- **Single-threaded executor**: Processes callbacks sequentially
- **Multi-threaded executor**: Processes callbacks in parallel using a thread pool
- **Custom executors**: For specialized concurrency patterns

## Practical Implementation

### Setting Up Python Environment

Before creating ROS 2 Python nodes, you need to set up your environment:

1. Source the ROS 2 environment:
```bash
source /opt/ros/humble/setup.bash  # Replace with your ROS 2 distribution
```

2. Create a Python package with proper dependencies:
```bash
mkdir ~/ros2_workspace/src/my_robot_package
cd ~/ros2_workspace/src/my_robot_package
```

3. Create setup.py and package.xml files with required dependencies

### Basic Node Structure

Here's the basic structure of a ROS 2 Python node:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('my_robot_node')
        
        # Log a message
        self.get_logger().info('MyRobotNode initialized')

def main():
    # Initialize rclpy
    rclpy.init()
    
    # Create node instance
    node = MyRobotNode()
    
    try:
        # Spin the node (process callbacks)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Implementing Publishers

Here's how to create and use publishers in rclpy:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Twist

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        
        # Create multiple publishers
        self.string_publisher = self.create_publisher(
            String,
            'status',
            10  # QoS queue size
        )
        
        self.count_publisher = self.create_publisher(
            Int32,
            'counter',
            10
        )
        
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        # Create a timer to publish messages at regular intervals
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        
    def timer_callback(self):
        # Publish string message
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.string_publisher.publish(msg)
        
        # Publish counter message
        count_msg = Int32()
        count_msg.data = self.i
        self.count_publisher.publish(count_msg)
        
        # Publish velocity command
        vel_msg = Twist()
        vel_msg.linear.x = 0.1  # Move forward slowly
        self.cmd_vel_publisher.publish(vel_msg)
        
        self.get_logger().info(f'Publishing: "{msg.data}", counter: {count_msg.data}')
        self.i += 1

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

### Implementing Subscribers

Here's how to create and use subscribers in rclpy:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import math

class DataProcessor(Node):
    def __init__(self):
        super().__init__('data_processor')
        
        # Create subscribers
        self.string_subscription = self.create_subscription(
            String,
            'status',
            self.string_callback,
            10
        )
        
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )
        
        # Prevent unused variable warning
        self.string_subscription
        self.scan_subscription
        
        # Initialize data storage
        self.last_status = None
        self.obstacle_distances = []
        
    def string_callback(self, msg):
        self.last_status = msg.data
        self.get_logger().info(f'Received status: {msg.data}')
        
    def scan_callback(self, msg):
        # Process laser scan data
        min_distance = float('inf')
        for range_val in msg.ranges:
            if not math.isnan(range_val) and range_val < min_distance:
                min_distance = range_val
                
        if min_distance < 1.0:  # Obstacle within 1 meter
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')
        else:
            self.get_logger().info(f'Clear path, nearest obstacle at {min_distance:.2f}m')

def main():
    rclpy.init()
    node = DataProcessor()
    
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

### Implementing Services

Here's how to create and use services in rclpy:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class CalculatorService(Node):
    def __init__(self):
        super().__init__('calculator_service')
        
        # Create service server
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )
        
    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response

def main():
    rclpy.init()
    node = CalculatorService()
    
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

### Multi-threaded Execution

For nodes that need to handle multiple callbacks concurrently:

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
import time

class ThreadingExample(Node):
    def __init__(self):
        super().__init__('threading_example')
        
        # Publisher
        self.publisher = self.create_publisher(String, 'thread_status', 10)
        
        # Timers with different periods
        self.create_timer(1.0, self.slow_timer_callback)
        self.create_timer(0.1, self.fast_timer_callback)
        
    def slow_timer_callback(self):
        self.get_logger().info('Slow timer executing...')
        # Simulate slow operation
        time.sleep(0.5)  # This would block single-threaded executor
        msg = String()
        msg.data = 'Slow operation completed'
        self.publisher.publish(msg)
        
    def fast_timer_callback(self):
        self.get_logger().info('Fast timer executing...')
        msg = String()
        msg.data = 'Fast update'
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = ThreadingExample()
    
    # Use multi-threaded executor to handle blocking operations
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

### Parameters in rclpy

ROS 2 nodes can use parameters for configuration:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

class ParameterExample(Node):
    def __init__(self):
        super().__init__('parameter_example')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value

        self.get_logger().info(
            f'Robot: {self.robot_name}, Max vel: {self.max_velocity}, '
            f'Safety dist: {self.safety_distance}'
        )

        # Create callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                if 0.0 < param.value <= 5.0:  # Validate parameter
                    self.max_velocity = param.value
                    self.get_logger().info(f'Max velocity updated to {param.value}')
                    return SetParametersResult(successful=True)
                else:
                    self.get_logger().error(f'Invalid max_velocity: {param.value}')
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

def main():
    rclpy.init()
    node = ParameterExample()

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

### Complete Practical Example: TurtleBot Navigation System

Here's a comprehensive example that combines multiple ROS 2 concepts to create a simple navigation system for a TurtleBot-like robot:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
import math
import time

class TurtleBotNavigator(Node):
    def __init__(self):
        super().__init__('turtlebot_navigator')

        # Declare parameters
        self.declare_parameter('linear_speed', 0.2)  # m/s
        self.declare_parameter('angular_speed', 0.5)  # rad/s
        self.declare_parameter('safety_distance', 0.5)  # meters
        self.declare_parameter('target_distance', 2.0)  # meters

        # Get parameters
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.target_distance = self.get_parameter('target_distance').value

        # Create publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Create subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.status_publisher = self.create_publisher(
            String,
            '/navigation_status',
            10
        )

        # Navigation state
        self.laser_ranges = []
        self.position = Point()
        self.orientation = 0.0
        self.nav_state = "idle"  # idle, moving, avoiding, reached
        self.start_time = time.time()
        self.distance_traveled = 0.0
        self.last_position = Point()

        # Timer for navigation logic
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)

        self.get_logger().info('TurtleBot Navigator initialized')

    def scan_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        self.laser_ranges = msg.ranges

        # Find the minimum distance in front of the robot
        front_scan = self.laser_ranges[len(self.laser_ranges)//2 - 90:len(self.laser_ranges)//2 + 90]
        if front_scan:
            self.min_front_distance = min([r for r in front_scan if not math.isnan(r)], default=float('inf'))
        else:
            self.min_front_distance = float('inf')

    def odom_callback(self, msg):
        """Process odometry data to track position"""
        self.position.x = msg.pose.pose.position.x
        self.position.y = msg.pose.pose.position.y
        self.position.z = msg.pose.pose.position.z

        # Simple conversion of quaternion to yaw (for 2D navigation)
        # In a real system, you might want to use tf2 for this
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.orientation = math.atan2(siny_cosp, cosy_cosp)

        # Calculate distance traveled since last check
        if self.last_position.x != 0 or self.last_position.y != 0:
            self.distance_traveled += math.sqrt(
                (self.position.x - self.last_position.x)**2 +
                (self.position.y - self.last_position.y)**2
            )

        self.last_position.x = self.position.x
        self.last_position.y = self.position.y

    def navigation_loop(self):
        """Main navigation loop"""
        cmd_msg = Twist()

        if self.nav_state == "idle":
            # Start navigation
            self.nav_state = "moving"
            self.start_time = time.time()
            self.distance_traveled = 0.0
            self.last_position = Point()
            cmd_msg.linear.x = self.linear_speed
            cmd_msg.angular.z = 0.0

        elif self.nav_state == "moving":
            # Check if obstacle is detected
            if self.min_front_distance <= self.safety_distance:
                self.nav_state = "avoiding"
                self.get_logger().info('Obstacle detected, starting avoidance')
            # Check if target distance is reached
            elif self.distance_traveled >= self.target_distance:
                self.nav_state = "reached"
                self.get_logger().info('Target distance reached')
            else:
                # Continue moving forward
                cmd_msg.linear.x = self.linear_speed
                cmd_msg.angular.z = 0.0

        elif self.nav_state == "avoiding":
            # Simple obstacle avoidance - turn right
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = -self.angular_speed

            # Resume forward motion if obstacle is cleared
            if self.min_front_distance > self.safety_distance * 1.5:
                self.nav_state = "moving"
                cmd_msg.linear.x = self.linear_speed
                cmd_msg.angular.z = 0.0

        elif self.nav_state == "reached":
            # Stop the robot
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0

            # Publish status
            status_msg = String()
            status_msg.data = f"Target reached after {time.time() - self.start_time:.1f}s and {self.distance_traveled:.2f}m"
            self.status_publisher.publish(status_msg)

            # Return to idle after a delay
            if time.time() - self.start_time > 2.0:
                self.nav_state = "idle"
                self.get_logger().info('Navigation cycle completed, returning to idle')

        # Send velocity command
        self.cmd_vel_publisher.publish(cmd_msg)

        # Log current status
        status_msg = String()
        status_msg.data = f"State: {self.nav_state}, Distance: {self.distance_traveled:.2f}m, Obstacle: {self.min_front_distance:.2f}m"
        self.status_publisher.publish(status_msg)

def main():
    rclpy.init()
    navigator = TurtleBotNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This comprehensive example demonstrates:
- Using multiple message types (Twist, LaserScan, Odometry, String)
- Parameter management for configuration
- Multiple publishers and subscribers
- State management for navigation
- Real-time processing of sensor data
- Integration of perception (LaserScan) with action (Twist)

## Best Practices

### Error Handling and Logging

```python
import rclpy
from rclpy.node import Node

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')
        
        # Proper error handling
        try:
            self.setup_components()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize: {str(e)}')
            raise
    
    def setup_components(self):
        # Setup publishers, subscribers, etc.
        pass
```

### Resource Management

```python
class ResourceManagingNode(Node):
    def __init__(self):
        super().__init__('resource_node')
        self.resources = []
        
    def destroy_node(self):
        # Clean up resources
        for resource in self.resources:
            if hasattr(resource, 'cleanup'):
                resource.cleanup()
        super().destroy_node()
```

## Interactive Exercises

### Exercise 1: Time Publisher and Subscriber
**Objective**: Create a publisher-subscriber pair that communicates the current time.

1. Create a publisher node that publishes the current timestamp to a `/current_time` topic at 1Hz
2. Create a subscriber node that listens to this topic and prints the received time
3. Add a parameter to control the publishing frequency
4. **Extension**: Have the subscriber calculate and display the time difference between consecutive messages

**Solution Steps**:
```
# Create a package:
# ros2 pkg create --build-type ament_python time_publisher_subscriber
# Add the publisher and subscriber code in separate files
```

### Exercise 2: Distance Calculation Service
**Objective**: Create a service that calculates the distance between two points.

1. Create a service server that accepts two coordinate points (x1, y1, x2, y2) and returns the Euclidean distance
2. Create a client that sends different coordinate pairs to the service
3. Test the service with multiple coordinate pairs
4. **Extension**: Add error handling for invalid input values

**Solution Steps**:
```
# Create a service definition file (in srv folder)
# Implement the service server and client
# Use the example_interfaces/srv/AddTwoInts as a template
```

### Exercise 3: Configurable Robot Controller
**Objective**: Create a node with parameters that control robot behavior.

1. Create a node that accepts parameters for robot speed (linear and angular), safety distance, and target distance
2. Implement a simple navigation behavior that uses these parameters
3. Test changing parameters at runtime using ros2 param commands
4. **Extension**: Add validation in the parameter callback to ensure parameters are within safe ranges

**Solution Steps**:
```
# Use self.declare_parameter() to declare parameters
# Use self.get_parameter() to access parameter values
# Add parameter callback to handle runtime updates
```

### Exercise 4: Sensor Processing Node
**Objective**: Create a node that processes sensor data and makes decisions.

1. Create a publisher that sends simulated sensor data (e.g., temperature readings)
2. Create a subscriber that processes this data and makes decisions (e.g., "normal", "warning", "danger")
3. Publish the decision to another topic
4. Create a third node that subscribes to the decision topic and logs actions
5. **Extension**: Add statistical analysis (mean, median, variance) of the sensor readings

**Solution Steps**:
```
# Use std_msgs for basic data types
# Implement callbacks to process incoming data
# Use create_timer() for periodic processing if needed
```

### Exercise 5: Multi-Node Communication System
**Objective**: Create a system with multiple interconnected nodes.

1. Create a sensor node that publishes random sensor readings
2. Create a filter node that receives sensor data, applies a simple filter, and republishes
3. Create a monitor node that receives the filtered data and alerts if values exceed thresholds
4. Create a logger node that records all sensor readings
5. Use rqt_graph to visualize the node connections
6. **Extension**: Add services to change threshold values dynamically

**Solution Steps**:
```
# Each node should focus on a single responsibility
# Use appropriate message types (or create custom ones)
# Test with ros2 run and examine node connections with rqt_graph
```

### Exercise 6: Parameter-Based Behavior Control
**Objective**: Create a robot controller that changes behavior based on parameters.

1. Create a parameter server node (or use parameters on your main node) with settings for robot behavior modes
2. Allow switching between different behaviors via parameters (e.g., patrol, follow, avoid)
3. Demonstrate how parameters can alter the execution without restarting the node
4. **Extension**: Create an action server that can execute long-running behaviors with feedback

**Solution Steps**:
```
# Use self.add_on_set_parameters_callback() to handle parameter changes
# Implement state machine to handle different behaviors
# Use ros2 param set commands to change behavior at runtime
```

### Exercise 7: System Integration Challenge
**Objective**: Combine multiple concepts into a comprehensive system.

1. Create a robot simulation with multiple sensors (e.g., laser range finder, camera simulator)
2. Implement nodes for each sensor processing
3. Create a central decision-making node that integrates all sensor data
4. Add a navigation controller that responds to the decision maker
5. Use services for high-level commands (start, stop, return to base)
6. Use parameters to configure the behavior of the entire system
7. **Extension**: Add a visualization node that displays system state

**Solution Steps**:
```
# Plan your node architecture before implementing
# Use appropriate message types for inter-node communication
# Test components individually before system integration
# Use rqt tools to monitor system performance
```

### Tools for Testing and Verification

To test your exercises, you can use these tools:
- `ros2 run <package> <node>` to run individual nodes
- `ros2 topic list` and `ros2 topic echo <topic_name>` to examine topics
- `ros2 service list` and `ros2 service call <service_name> <type> '{request: ...}'` to test services
- `rqt_graph` to visualize the node graph
- `ros2 param list` and `ros2 param get <node_name> <param_name>` to check parameters
- `ros2 launch <package> <launch_file>` to run multiple nodes together

## Hands-on Package Creation Exercises

### Exercise A: Creating Your First ROS 2 Package

1. **Initialize a Workspace**
   ```bash
   mkdir -p ~/ros2_workspace/src
   cd ~/ros2_workspace/src
   ```

2. **Create the Package**
   ```bash
   ros2 pkg create --build-type ament_python first_robot_package
   cd first_robot_package
   ```

3. **File Structure Created**
   ```
   first_robot_package/
   ├── first_robot_package/
   │   ├── __init__.py
   │   └── simple_publisher.py  # We'll create this
   ├── package.xml
   ├── setup.cfg
   ├── setup.py
   └── test/
       ├── __init__.py
       ├── test_copyright.py
       ├── test_flake8.py
       └── test_pep257.py
   ```

4. **Create a Simple Publisher Node**
   Create `first_robot_package/simple_publisher.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class SimplePublisher(Node):
       def __init__(self):
           super().__init__('simple_publisher')
           self.publisher = self.create_publisher(String, 'chatter', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Hello World: {self.i}'
           self.publisher.publish(msg)
           self.get_logger().info(f'Publishing: "{msg.data}"')
           self.i += 1

   def main():
       rclpy.init()
       node = SimplePublisher()

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

5. **Set Execute Permissions**
   ```bash
   chmod +x first_robot_package/simple_publisher.py
   ```

6. **Modify setup.py**
   Add the following to the setup() call in setup.py:
   ```python
   entry_points={
       'console_scripts': [
           'simple_publisher = first_robot_package.simple_publisher:main',
       ],
   },
   ```

7. **Build the Package**
   ```bash
   cd ~/ros2_workspace
   colcon build --packages-select first_robot_package
   ```

8. **Source the Workspace**
   ```bash
   source install/setup.bash
   ```

9. **Run the Node**
   ```bash
   ros2 run first_robot_package simple_publisher
   ```

### Exercise B: Creating a Publisher-Subscriber Package

1. **Create a New Package**
   ```bash
   cd ~/ros2_workspace/src
   ros2 pkg create --build-type ament_python comm_package
   cd comm_package
   ```

2. **Create Publisher Node**
   Create `comm_package/talker.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class Talker(Node):
       def __init__(self):
           super().__init__('talker')
           self.publisher = self.create_publisher(String, 'messages', 10)
           timer_period = 0.5
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.counter = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Hello from talker: {self.counter}'
           self.publisher.publish(msg)
           self.get_logger().info(f'Publishing: {msg.data}')
           self.counter += 1

   def main():
       rclpy.init()
       talker = Talker()

       try:
           rclpy.spin(talker)
       except KeyboardInterrupt:
           pass
       finally:
           talker.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create Subscriber Node**
   Create `comm_package/listener.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class Listener(Node):
       def __init__(self):
           super().__init__('listener')
           self.subscription = self.create_subscription(
               String,
               'messages',
               self.listener_callback,
               10)
           self.subscription  # Prevent unused variable warning

       def listener_callback(self, msg):
           self.get_logger().info(f'I heard: "{msg.data}"')

   def main():
       rclpy.init()
       listener = Listener()

       try:
           rclpy.spin(listener)
       except KeyboardInterrupt:
           pass
       finally:
           listener.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Update setup.py**
   Add both nodes to console_scripts in setup.py:
   ```python
   entry_points={
       'console_scripts': [
           'talker = comm_package.talker:main',
           'listener = comm_package.listener:main',
       ],
   },
   ```

5. **Build and Run**
   ```bash
   cd ~/ros2_workspace
   colcon build --packages-select comm_package
   source install/setup.bash

   # Terminal 1:
   ros2 run comm_package talker

   # Terminal 2:
   ros2 run comm_package listener
   ```

### Exercise C: Creating a Service Server and Client Package

1. **Create Package**
   ```bash
   cd ~/ros2_workspace/src
   ros2 pkg create --build-type ament_python service_demo
   cd service_demo
   ```

2. **Create Service Server**
   Create `service_demo/add_two_ints_server.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from example_interfaces.srv import AddTwoInts

   class AddTwoIntsServer(Node):
       def __init__(self):
           super().__init__('add_two_ints_server')
           self.srv = self.create_service(
               AddTwoInts,
               'add_two_ints',
               self.add_two_ints_callback)

       def add_two_ints_callback(self, request, response):
           response.sum = request.a + request.b
           self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
           return response

   def main():
       rclpy.init()
       server = AddTwoIntsServer()

       try:
           rclpy.spin(server)
       except KeyboardInterrupt:
           pass
       finally:
           server.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create Service Client**
   Create `service_demo/add_two_ints_client.py`:
   ```python
   #!/usr/bin/env python3
   import sys
   import rclpy
   from rclpy.node import Node
   from example_interfaces.srv import AddTwoInts

   class AddTwoIntsClient(Node):
       def __init__(self):
           super().__init__('add_two_ints_client')
           self.cli = self.create_client(AddTwoInts, 'add_two_ints')
           while not self.cli.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Service not available, waiting again...')
           self.req = AddTwoInts.Request()

       def send_request(self, a, b):
           self.req.a = a
           self.req.b = b
           self.future = self.cli.call_async(self.req)
           rclpy.spin_until_future_complete(self, self.future)
           return self.future.result()

   def main():
       rclpy.init()
       client = AddTwoIntsClient()

       if len(sys.argv) != 3:
           print('Usage: ros2 run service_demo add_two_ints_client X Y')
           return

       a = int(sys.argv[1])
       b = int(sys.argv[2])

       response = client.send_request(a, b)
       if response:
           print(f'Result of {a} + {b} = {response.sum}')
       else:
           print('Service call failed')

       client.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Update setup.py**
   ```python
   entry_points={
       'console_scripts': [
           'add_two_ints_server = service_demo.add_two_ints_server:main',
           'add_two_ints_client = service_demo.add_two_ints_client:main',
       ],
   },
   ```

5. **Build and Test**
   ```bash
   cd ~/ros2_workspace
   colcon build --packages-select service_demo
   source install/setup.bash

   # Terminal 1:
   ros2 run service_demo add_two_ints_server

   # Terminal 2:
   ros2 run service_demo add_two_ints_client 2 3
   ```

### Exercise D: Using Parameters in Your Package

1. **Create Parameter Node**
   ```bash
   cd ~/ros2_workspace/src/comm_package
   # Create comm_package/parameter_node.py
   ```

   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class ParameterNode(Node):
       def __init__(self):
           super().__init__('parameter_node')

           # Declare parameters with default values
           self.declare_parameter('message_prefix', 'Robot says: ')
           self.declare_parameter('publish_rate', 1.0)
           self.declare_parameter('message_count', 10)

           # Get parameter values
           self.prefix = self.get_parameter('message_prefix').value
           self.rate = self.get_parameter('publish_rate').value
           self.count_limit = self.get_parameter('message_count').value

           # Create publisher
           self.publisher = self.create_publisher(String, 'param_messages', 10)

           # Create timer based on parameter
           self.timer = self.create_timer(1.0/self.rate, self.timer_callback)
           self.counter = 0

       def timer_callback(self):
           if self.counter < self.count_limit:
               msg = String()
               msg.data = f'{self.prefix}Message #{self.counter}'
               self.publisher.publish(msg)
               self.get_logger().info(f'Published: {msg.data}')
               self.counter += 1
           else:
               self.get_logger().info('Reached message count limit, stopping.')
               # In a real system, you might want to stop the node here

   def main():
       rclpy.init()
       node = ParameterNode()

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

2. **Update setup.py**
   ```python
   entry_points={
       'console_scripts': [
           'talker = comm_package.talker:main',
           'listener = comm_package.listener:main',
           'parameter_node = comm_package.parameter_node:main',
       ],
   },
   ```

3. **Build and Test with Parameters**
   ```bash
   cd ~/ros2_workspace
   colcon build --packages-select comm_package
   source install/setup.bash

   # Run with default parameters
   ros2 run comm_package parameter_node

   # Run with custom parameters
   ros2 run comm_package parameter_node --ros-args -p message_prefix:="Custom prefix: " -p publish_rate:=2.0
   ```

### Exercise E: Creating a Launch File for Multiple Nodes

1. **Create Launch Directory**
   ```bash
   mkdir -p ~/ros2_workspace/src/comm_package/launch
   ```

2. **Create Launch File**
   Create `~/ros2_workspace/src/comm_package/launch/communication_nodes.launch.py`:
   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node

   def generate_launch_description():
       return LaunchDescription([
           Node(
               package='comm_package',
               executable='talker',
               name='talker_node',
               parameters=[
                   {'publish_rate': 2.0}  # Example parameter override
               ],
               output='screen'
           ),
           Node(
               package='comm_package',
               executable='listener',
               name='listener_node',
               output='screen'
           ),
           Node(
               package='comm_package',
               executable='parameter_node',
               name='param_node',
               parameters=[
                   {'message_prefix': 'Launch file: '}
               ],
               output='screen'
           )
       ])
   ```

3. **Update setup.py to include launch files**
   Modify the data_files section in setup.py:
   ```python
   import os
   from glob import glob
   from setuptools import setup

   # ... existing setup code ...

   setup(
       # ... other setup parameters ...
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/comm_package']),
           ('share/comm_package',
            ['package.xml']),
           # Include launch files
           (os.path.join('share', 'comm_package', 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
       ],
       # ... rest of setup parameters ...
   )
   ```

4. **Build and Run with Launch File**
   ```bash
   cd ~/ros2_workspace
   colcon build --packages-select comm_package
   source install/setup.bash

   ros2 launch comm_package communication_nodes.launch.py
   ```

### Exercise F: Creating Custom Message Types

1. **Create Custom Message Package**
   ```bash
   cd ~/ros2_workspace/src
   ros2 pkg create --build-type ament_python custom_msgs --dependencies std_msgs builtin_interfaces
   cd custom_msgs
   ```

2. **Create Message Definition**
   ```bash
   mkdir msg
   # Create msg/NumArray.msg
   ```

   Content of `msg/NumArray.msg`:
   ```
   # Custom message for an array of numbers
   float64[] data
   string description
   builtin_interfaces/Time timestamp
   ```

3. **Update package.xml**
   Add dependencies for message generation:
   ```xml
   <buildtool_depend>ament_python</buildtool_depend>
   <depend>std_msgs</depend>
   <depend>builtin_interfaces</depend>
   <member_of_group>rosidl_interface_packages</member_of_group>
   ```

4. **Update setup.py**
   ```python
   from setuptools import setup
   from glob import glob
   import os

   package_name = 'custom_msgs'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Custom messages for Physical AI textbook',
       license='MIT',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
           ],
       },
       # Add the following for custom messages
       packages=[package_name],
       py_modules=[],
       package_data={},
       # This is important for message generation
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name,
            ['package.xml']),
           ('share/' + package_name + '/msg',
            ['msg/NumArray.msg']),
       ],
       # Add this to generate messages
       # The exact configuration would depend on your ROS 2 distribution
       # In newer versions, you might need to use:
       # 'rosidl_generate_interfaces': ['custom_msgs'],
   )
   ```

5. **Build and Use Custom Message**
   ```bash
   cd ~/ros2_workspace
   colcon build --packages-select custom_msgs
   source install/setup.bash
   ```

These exercises provide hands-on experience creating complete ROS 2 packages with various node types, parameters, launch files, and custom messages.

## Troubleshooting Common ROS 2 Issues

### Common Issues and Solutions

#### 1. Module Import Errors
**Problem**: `ModuleNotFoundError: No module named 'rclpy'`
**Solution**:
- Ensure ROS 2 is sourced: `source /opt/ros/<distro>/setup.bash`
- Install ROS 2 Python packages if missing
- Check if you're using the correct Python environment

#### 2. Node Connection Issues
**Problem**: Nodes can't communicate with each other
**Solution**:
- Verify that both nodes use the same RMW (ROS Middleware) implementation
- Check if nodes are on the same ROS domain: `echo $ROS_DOMAIN_ID`
- Ensure DDS communication is not blocked by firewall
- Use `ros2 topic list` to confirm topics exist

#### 3. Topic/Service Names Conflicts
**Problem**: Nodes not communicating on expected topics/services
**Solution**:
- Use `ros2 node list` and `ros2 topic list` to see active entities
- Check for typos in topic/service names
- Verify QoS profiles match between publisher and subscriber
- Use `ros2 topic info <topic_name>` to see endpoint details

#### 4. Parameter Issues
**Problem**: Parameters not being read or updated at runtime
**Solution**:
- Ensure parameters are declared with `self.declare_parameter()`
- Use `ros2 param list <node_name>` to confirm parameters exist
- Check parameter names for typos
- Verify parameter callback is properly registered

#### 5. Resource Management Problems
**Problem**: Nodes not shutting down cleanly or resource leaks
**Solution**:
- Always call `node.destroy_node()` and `rclpy.shutdown()` in finally blocks
- Implement proper signal handling for Ctrl+C
- Check for circular references that might prevent garbage collection

#### 6. Threading and Concurrency Issues
**Problem**: Deadlocks or race conditions in multi-threaded nodes
**Solution**:
- Use MultiThreadedExecutor carefully
- Protect shared resources with locks
- Avoid blocking operations in callbacks when using SingleThreadedExecutor
- Consider using action servers for long-running tasks instead of blocking callbacks

#### 7. Performance Issues
**Problem**: Nodes running slowly or missing messages
**Solution**:
- Check queue sizes in publisher/subscriber creation
- Reduce unnecessary logging in high-frequency callbacks
- Profile node performance with `ros2 run topic_tools delay <topic>`
- Consider using different QoS profiles (e.g., durability, reliability settings)

#### 8. Package and Build Issues
**Problem**: Nodes not running after building with colcon
**Solution**:
- Source the install directory: `source install/setup.bash`
- Ensure console_scripts are properly defined in setup.py
- Check executable permissions on Python files
- Verify package.xml includes all necessary dependencies

### Debugging Techniques

#### 1. Verbose Logging
Always include logging to help debug issues:
```python
self.get_logger().info('Detailed information about current state')
self.get_logger().warn('Warning about potential issues')
self.get_logger().error('Error conditions that need addressing')
```

#### 2. Using ROS 2 Command-line Tools
```bash
# Monitor nodes
ros2 node list
ros2 node info <node_name>

# Monitor topics
ros2 topic list
ros2 topic echo <topic_name>
ros2 topic info <topic_name>

# Monitor services
ros2 service list
ros2 service type <service_name>

# Monitor parameters
ros2 param list <node_name>
ros2 param get <node_name> <param_name>
```

#### 3. Runtime Parameter Adjustment
Test different parameter values during runtime:
```bash
# Change parameter while node is running
ros2 param set <node_name> <param_name> <value>

# Example:
ros2 param set my_node publish_rate 2.0
```

#### 4. Visualization Tools
Use rqt tools for debugging:
```bash
# Node graph visualization
rqt_graph

# Message monitoring
rqt_plot
rqt_console
rqt_bag
```

### Best Practices for Avoiding Common Issues

1. **Always use proper exception handling**:
```python
def main():
    rclpy.init()
    node = MyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

2. **Validate incoming messages**:
```python
def callback(self, msg):
    # Check for invalid values
    if not self.is_valid_message(msg):
        self.get_logger().error('Received invalid message')
        return
    # Process valid message
    self.process_message(msg)
```

3. **Use appropriate QoS profiles** for your application:
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# For sensor data (frequent updates, can lose some messages)
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)

# For critical commands (must be delivered)
qos_profile = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
```

4. **Plan for graceful degradation** when optional components fail:
```python
class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')
        self.vision_available = True
        self.laser_available = True

        # Try to create optional subscribers
        try:
            self.vision_sub = self.create_subscription(...)
        except Exception as e:
            self.get_logger().warn(f'Vision not available: {e}')
            self.vision_available = False
```

## Key Takeaways

- rclpy provides the Python interface to ROS 2 functionality
- Python nodes inherit from rclpy.node.Node class
- Publishers, subscribers, services, and clients are created through node methods
- Executors manage concurrency in ROS 2 Python nodes
- Parameters provide configuration management
- Proper resource management and error handling are important in production systems

## Further Reading

- ROS 2 Python Developer Guide
- "Effective Robotics Programming with ROS" by Anil Mahtani
- Python rclpy API documentation

## Next Steps

Continue to Module 2, Part 2: URDF Format to learn about robot description and modeling.