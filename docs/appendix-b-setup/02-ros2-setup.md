---
sidebar_position: 2
title: ROS 2 Setup
---

# ROS 2 Setup for Physical AI Development

This chapter provides detailed instructions for configuring ROS 2 specifically for Physical AI and humanoid robotics applications. Proper ROS 2 configuration is essential for effective robotics development and will be used throughout this textbook.

## Learning Objectives

- Configure ROS 2 environment for optimal robotics development
- Set up ROS 2 workspaces and packages for robotics projects
- Configure ROS 2 for multi-robot simulation and control
- Understand ROS 2 networking and performance considerations
- Implement ROS 2 best practices for Physical AI applications

## Introduction: Why ROS 2 Configuration Matters

ROS 2 (Robot Operating System 2) is the communication middleware that connects all components of a robotic system. Proper configuration affects:

- **Performance**: How quickly messages pass between components
- **Reliability**: How robustly the system handles communication failures
- **Scalability**: How well the system handles multiple robots or complex systems
- **Security**: How protected the system is from unauthorized access
- **Real-time behavior**: How predictable the system is for time-critical operations

For Physical AI, ROS 2 configuration becomes particularly important because physical robots must operate in real-time with strict timing requirements and safety considerations.

## ROS 2 Environment Configuration

### Setting Up the ROS 2 Environment

First, ensure the ROS 2 environment is properly sourced. Add these lines to your `~/.bashrc` if not already there:

```bash
# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Source your workspace (assumes you created ~/ros2_ws as in the previous chapter)
source ~/ros2_ws/install/setup.bash

# Set ROS 2 environment variables for robotics development
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
export ROS_DOMAIN_ID=42  # Change for multi-robot systems
export ROS_LOCALHOST_ONLY=0  # Allow network communication
```

### Understanding RMW Implementations

ROS 2 uses different middleware implementations. For Physical AI applications:

- **rmw_cyclonedx_cpp** (Default): Good balance of performance and features
- **rmw_fastrtps_cpp**: Higher performance for time-critical applications
- **rmw_connextdds**: For high-performance applications (requires license)

To set the implementation:
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
```

### Configuring Domain ID

ROS 2 uses domain IDs to separate different ROS 2 networks:

```bash
# Default domain (0-255, 0-23 typically used by other applications)
export ROS_DOMAIN_ID=42

# For multiple robot systems, use different domains:
# Robot 1: export ROS_DOMAIN_ID=42
# Robot 2: export ROS_DOMAIN_ID=43
# Shared network: export ROS_DOMAIN_ID=44
```

## Creating a Robotics Development Workspace

### Workspace Structure for Physical AI

For Physical AI projects, we recommend a structured workspace:

```
~/physical_ai_ws/
├── src/
│   ├── robot_models/         # URDF models, meshes, Gazebo worlds
│   ├── control_packages/     # Controllers and motion planning
│   ├── perception_packages/  # Vision, sensors, object detection
│   ├── simulation_packages/  # Gazebo plugins, simulation worlds
│   └── application_packages/# High-level behaviors, AI applications
├── install/
├── build/
└── log/
```

### Creating the Workspace

```bash
# Create the workspace
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws

# Create subdirectories for organization
mkdir -p src/{robot_models,control_packages,perception_packages,simulation_packages,application_packages}

# Initialize the workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Adding Environment Configuration

Add workspace-specific environment variables to `~/.bashrc`:

```bash
# Physical AI Development Workspace
export PHYSICAL_AI_WS=~/physical_ai_ws
export GZ_SIM_RESOURCE_PATH=$PHYSICAL_AI_WS/src/robot_models:$GZ_SIM_RESOURCE_PATH
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PHYSICAL_AI_WS/install/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH
export AMENT_PREFIX_PATH=$PHYSICAL_AI_WS/install:$AMENT_PREFIX_PATH
export CMAKE_PREFIX_PATH=$PHYSICAL_AI_WS/install:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$PHYSICAL_AI_WS/install/lib:$LD_LIBRARY_PATH
export PATH=$PHYSICAL_AI_WS/install/bin:$PATH
export PKG_CONFIG_PATH=$PHYSICAL_AI_WS/install/lib/pkgconfig:$PKG_CONFIG_PATH
export PYTHONPATH=$PHYSICAL_AI_WS/install/lib/python3.10/site-packages:$PYTHONPATH
export ROS_PACKAGE_PATH=$PHYSICAL_AI_WS/src:$ROS_PACKAGE_PATH
```

## ROS 2 Quality of Service (QoS) Configuration

Proper QoS configuration is critical for Physical AI applications, especially for safety and real-time behavior:

### Understanding QoS Settings

```python
import rclpy.qos as qos

# Reliability: Can messages be dropped?
reliable = qos.ReliabilityPolicy.RELIABLE          # All messages delivered
best_effort = qos.ReliabilityPolicy.BEST_EFFORT    # Try to deliver messages

# Durability: Should messages persist?
volatile = qos.DurabilityPolicy.VOLATILE           # Only new subscribers get messages
transient_local = qos.DurabilityPolicy.TRANSIENT_LOCAL  # All subscribers get last message

# History: How many messages to keep?
keep_last = qos.HistoryPolicy.KEEP_LAST            # Keep N most recent messages
keep_all = qos.HistoryPolicy.KEEP_ALL             # Keep all messages
```

### Recommended QoS for Different Message Types

```python
# For safety-critical messages (e.g., motor commands)
safety_qos = qos.QoSProfile(
    depth=1,
    reliability=qos.ReliabilityPolicy.RELIABLE,
    durability=qos.DurabilityPolicy.VOLATILE,
    history=qos.HistoryPolicy.KEEP_LAST
)

# For sensor data (e.g., camera feeds, LiDAR)
sensor_qos = qos.QoSProfile(
    depth=5,
    reliability=qos.ReliabilityPolicy.BEST_EFFORT,
    durability=qos.DurabilityPolicy.VOLATILE,
    history=qos.HistoryPolicy.KEEP_LAST
)

# For configuration parameters
config_qos = qos.QoSProfile(
    depth=1,
    reliability=qos.ReliabilityPolicy.RELIABLE,
    durability=qos.DurabilityPolicy.TRANSIENT_LOCAL,
    history=qos.HistoryPolicy.KEEP_LAST
)

# For diagnostics and logging
diagnostic_qos = qos.QoSProfile(
    depth=10,
    reliability=qos.ReliabilityPolicy.BEST_EFFORT,
    durability=qos.DurabilityPolicy.VOLATILE,
    history=qos.HistoryPolicy.KEEP_LAST
)
```

### Using QoS in Code

```python
class PhysicalRobotController(Node):
    def __init__(self):
        super().__init__('physical_robot_controller')
        
        # Publisher with safety-focused QoS
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            safety_qos  # Use the safety-focused QoS profile
        )
        
        # Publisher with sensor-focused QoS
        self.image_publisher = self.create_publisher(
            Image,
            'camera/image_raw',
            sensor_qos  # Use the sensor-focused QoS profile
        )
        
        # Subscriber for safety-critical commands
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            safety_qos
        )
```

## ROS 2 Parameters Configuration

Parameters allow runtime configuration of robot behavior:

### Setting Parameters for Physical AI

```python
class PhysicalController(Node):
    def __init__(self):
        super().__init__('physical_controller')
        
        # Declare parameters with default values
        self.declare_parameter('max_linear_velocity', 0.5)  # m/s
        self.declare_parameter('max_angular_velocity', 1.0) # rad/s
        self.declare_parameter('safety_distance', 0.5)      # meters
        self.declare_parameter('control_frequency', 50)     # Hz
        
        # Get parameter values
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.control_frequency = self.get_parameter('control_frequency').value
        
        # Create timer to run control loop at specified frequency
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop
        )
        
        self.get_logger().info(
            f'Controller initialized: max_linear_vel={self.max_linear_vel}, '
            f'max_angular_vel={self.max_angular_vel}, '
            f'safety_distance={self.safety_distance}, '
            f'control_frequency={self.control_frequency}'
        )
```

### Parameter Validation

```python
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

class ValidatedController(Node):
    def __init__(self):
        super().__init__('validated_controller')
        
        # Declare parameters with validation
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)
        
        # Add parameter callback to validate changes
        self.add_on_set_parameters_callback(self.parameter_callback)
    
    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_linear_velocity':
                if param.type_ == Parameter.Type.DOUBLE:
                    if 0.0 < param.value <= 5.0:
                        self.get_logger().info(f'Setting max_linear_velocity to {param.value}')
                        return SetParametersResult(successful=True)
                    else:
                        self.get_logger().error(
                            f'Invalid max_linear_velocity: {param.value}. '
                            f'Must be between 0.0 and 5.0'
                        )
                        return SetParametersResult(successful=False)
            elif param.name == 'max_angular_velocity':
                if param.type_ == Parameter.Type.DOUBLE:
                    if 0.0 < param.value <= 5.0:
                        self.get_logger().info(f'Setting max_angular_velocity to {param.value}')
                        return SetParametersResult(successful=True)
                    else:
                        self.get_logger().error(
                            f'Invalid max_angular_velocity: {param.value}. '
                            f'Must be between 0.0 and 5.0'
                        )
                        return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)
```

## ROS 2 Launch Files Configuration

Launch files are used to start multiple nodes together with specific configurations:

### Basic Launch File Structure

Create `~/physical_ai_ws/src/robot_models/launch/physical_robot.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Define launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='physical_robot',
        description='Name of the robot'
    )
    
    # Include Gazebo simulation
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'verbose': 'false'}.items(),
        condition=IfCondition(use_sim_time)
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(
                os.path.join(
                    get_package_share_directory('robot_models'),
                    'urdf',
                    'physical_robot.urdf'
                )
            ).read()
        }]
    )
    
    # Joint state publisher (for simulation)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        condition=IfCondition(use_sim_time)
    )
    
    # Physical robot controller
    robot_controller = Node(
        package='control_packages',
        executable='physical_robot_controller',
        name='physical_robot_controller',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot_name
        }],
        output='screen'
    )
    
    # Return the launch description
    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        robot_controller
    ])
```

## Multi-Robot Configuration

For Physical AI applications involving multiple robots:

### Network Configuration

```bash
# Create a network configuration file
# Create ~/physical_ai_ws/config/ros_network.yaml:
network:
  ros_domain_id: 42
  network_interface: eth0  # or wlan0 for wireless
  multicast_ttl: 32  # Increase for multi-hop networks
```

### Domain ID Management

For multi-robot systems, manage domain IDs carefully:

```python
import os
import random

class MultiRobotNode(Node):
    def __init__(self, robot_name):
        # Generate or use a specific domain ID
        domain_id = os.environ.get('ROS_DOMAIN_ID')
        if not domain_id:
            # Use robot-specific domain ID
            robot_hash = hash(robot_name) % 100
            domain_id = 42 + robot_hash
            os.environ['ROS_DOMAIN_ID'] = str(domain_id)
        
        super().__init__(f'{robot_name}_node')
        self.get_logger().info(f'Running on ROS_DOMAIN_ID={domain_id}')
```

## Performance Optimization Settings

### DDS Configuration for Performance

Create `~/physical_ai_ws/config/fastdds_profile.xml`:

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
    <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
        <transport_descriptors>
            <transport_descriptor>
                <transport_id>CustomUdpTransport</transport_id>
                <type>UDPv4</type>
                <sendBufferSize>65536</sendBufferSize>
                <receiveBufferSize>65536</receiveBufferSize>
            </transport_descriptor>
        </transport_descriptors>

        <participant profile_name="default_xrce_dds_participant" is_default_profile="true">
            <rtps>
                <userTransports>
                    <transport_id>CustomUdpTransport</transport_id>
                </userTransports>
                <useBuiltinTransports>false</useBuiltinTransports>
                <sendSocketBufferSize>65536</sendSocketBufferSize>
                <listenSocketBufferSize>65536</listenSocketBufferSize>
                <builtin>
                    <discovery_config>
                        <leaseDuration>
                            <sec>20</sec>
                        </leaseDuration>
                    </discovery_config>
                </builtin>
            </rtps>
        </participant>
    </profiles>
</dds>
```

### Real-time Configuration (Optional)

For hard real-time applications:

```bash
# Install real-time kernel (Ubuntu)
sudo apt install linux-image-rt-generic

# Add RT permissions to current user
echo "$USER soft rtprio 99" | sudo tee -a /etc/security/limits.conf
echo "$USER hard rtprio 99" | sudo tee -a /etc/security/limits.conf
echo "$USER soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "$USER hard memlock unlimited" | sudo tee -a /etc/security/limits.conf
```

## Testing ROS 2 Configuration

### Basic Functionality Test

```bash
# Test ROS 2 installation
ros2 run demo_nodes_cpp talker

# In another terminal
ros2 run demo_nodes_py listener

# Verify communication
echo "If messages appear in the listener terminal, ROS 2 is working"
```

### Network Test

```bash
# Test multi-machine communication
# On machine 1: export ROS_DOMAIN_ID=42; ros2 topic pub /test std_msgs/String "data: 'Hello from machine 1'"
# On machine 2: export ROS_DOMAIN_ID=42; ros2 topic echo /test
```

### Performance Test

```bash
# Test message latency
ros2 run demo_nodes_cpp add_two_ints_client_async

# Test QoS settings
ros2 topic echo /chatter sensor_msgs/msg/Image --field data --qos-profile sensor_data
```

## Troubleshooting Common ROS 2 Configuration Issues

### Issue 1: Nodes Can't Communicate

**Problem**: Nodes on the same machine can't see each other
**Solution**: Check domain ID and environment sourcing:

```bash
echo $ROS_DOMAIN_ID
echo $RMW_IMPLEMENTATION
# Verify both nodes have same values
```

### Issue 2: High Latency or Packet Loss

**Problem**: Messages are delayed or lost
**Solution**: Check network configuration:

```bash
# Test network performance
ping <target_machine>
# Check for network congestion
sudo apt install net-tools
netstat -i
```

### Issue 3: Parameter Service Unavailable

**Problem**: Cannot set parameters using ros2 param command
**Solution**: Ensure parameter services are enabled:

```python
# In your node, ensure you're using the standard node creation
# Parameters are automatically available in standard ROS 2 nodes
```

### Issue 4: Memory Issues with Large Messages

**Problem**: Large messages (like images) cause memory issues
**Solution**: Adjust QoS settings and use appropriate history depth:

```python
# For large messages, use smaller history depth
small_history_qos = qos.QoSProfile(
    depth=1,  # Keep only latest message
    reliability=qos.ReliabilityPolicy.BEST_EFFORT,
    durability=qos.DurabilityPolicy.VOLATILE,
    history=qos.HistoryPolicy.KEEP_LAST
)
```

## Security Configuration (Advanced)

For production Physical AI systems, security is important:

### ROS 2 Security Features

```bash
# Generate security files (advanced)
mkdir -p ~/physical_ai_ws/security
cd ~/physical_ai_ws/security

# Use ros2 security tools to generate keys and certificates
# This is for advanced users and production systems
# ros2 security create_keystore ~/physical_ai_ws/security
```

### Environment Variables for Security

```bash
# Enable security (requires security files)
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce
export ROS_SECURITY_KEYSTORE=~/physical_ai_ws/security
```

## Performance Monitoring

### Tools for Monitoring ROS 2 Performance

```bash
# Monitor topics
ros2 topic list
ros2 topic info /topic_name

# Monitor services
ros2 service list
ros2 service info /service_name

# Monitor nodes
ros2 node list

# Use rqt tools for visual monitoring
sudo apt install ros-humble-rqt ros-humble-rqt-common-plugins
rqt
```

## Containerized ROS 2 Setup (Docker)

For isolated development environments:

### Dockerfile for Physical AI Development

```dockerfile
FROM osrf/ros:humble-desktop-full

# Install additional packages for Physical AI
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install \
    numpy \
    matplotlib \
    opencv-python \
    transforms3d \
    pybullet

# Create workspace
RUN mkdir -p /root/physical_ai_ws/src
WORKDIR /root/physical_ai_ws

# Source ROS 2 environment
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /root/physical_ai_ws/install/setup.bash" >> /root/.bashrc

CMD ["bash"]
```

### Running ROS 2 in Docker with GUI

```bash
# For Linux with X11
xhost +local:docker
docker run -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --device=/dev/dri:/dev/dri:rw \
  --name=ros2_physical_ai \
  physical_ai_image:latest

# For Windows with WSL2 and VcXsrv
# Start VcXsrv with proper settings, then:
docker run -it \
  --env="DISPLAY=host.docker.internal:0.0" \
  physical_ai_image:latest
```

## Best Practices Summary

### Configuration Best Practices

1. **Environment Management**: Always source your complete environment before running ROS 2
2. **QoS Selection**: Choose appropriate QoS settings based on message criticality
3. **Parameter Validation**: Validate parameters to prevent runtime errors
4. **Domain Isolation**: Use different domain IDs for different robot systems
5. **Performance Monitoring**: Regularly monitor system performance
6. **Security**: Implement security measures for deployed systems
7. **Documentation**: Document all configuration parameters and settings

### Performance Best Practices

1. **Use Appropriate QoS**: Match QoS to message requirements
2. **Limit Message History**: Don't keep unnecessary message history
3. **Monitor Bandwidth**: Be aware of network bandwidth requirements
4. **Optimize Message Size**: Reduce message sizes when possible
5. **Use Efficient Data Types**: Choose appropriate data types for your needs

## Key Takeaways

- ROS 2 configuration significantly impacts Physical AI system performance
- Proper QoS settings are critical for safety and performance
- Workspace organization helps manage complex Physical AI projects
- Multi-robot systems require careful domain ID management
- Performance monitoring is essential for debugging and optimization
- Security considerations become important for deployed systems

## Further Reading

- "ROS 2 Design": Understanding the architecture of ROS 2
- "Quality of Service in ROS 2": Detailed QoS configuration guide
- "ROS 2 Performance Tuning": Optimizing ROS 2 applications
- "Multi-Robot Systems with ROS 2": Advanced configuration techniques

## Next Steps

Continue to Appendix B, Section 3: Gazebo Setup to configure the simulation environment for Physical AI development.