---
sidebar_position: 3
title: Gazebo Setup
---

# Gazebo Setup for Physical AI Simulation

This chapter provides comprehensive instructions for setting up Gazebo simulation environments for Physical AI and humanoid robotics applications. Gazebo is a critical tool for testing robotics algorithms in safe, repeatable environments before deployment to physical hardware.

## Learning Objectives

- Install and configure Gazebo Garden for Physical AI applications
- Create custom robot models and environments for simulation
- Integrate Gazebo with ROS 2 for seamless simulation-control workflows
- Understand physics parameters and their impact on robot behavior
- Troubleshoot common simulation issues and optimize performance

## Introduction: Simulation in Physical AI Development

Simulation is fundamental to Physical AI development, enabling:

- **Safe Testing**: Test algorithms without risk of damaging hardware
- **Repeatability**: Run experiments multiple times under identical conditions
- **Cost Efficiency**: Develop without expensive physical robots
- **Scalability**: Test multi-robot scenarios efficiently
- **Rapid Prototyping**: Quickly iterate on control and perception algorithms

For Physical AI applications, simulation must accurately model the physical world including:
- Physics: Gravity, friction, collision dynamics
- Sensors: Camera, LiDAR, IMU, force/torque sensing
- Environment: Lighting, textures, and physical interactions
- Real-time constraints: Proper timing for control systems

## Installing Gazebo Garden

### Prerequisites

Ensure your system meets Gazebo requirements:

- **Operating System**: Ubuntu 22.04 (recommended)
- **Graphics**: OpenGL 2.1+ compatible GPU with dedicated VRAM
- **Memory**: 8+ GB RAM (16+ GB recommended for complex scenes)
- **Storage**: 5+ GB free space

### Installation Methods

#### Method 1: Package Installation (Recommended)

```bash
# Add Gazebo package repository
curl -sSL http://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install gazebo-garden
```

#### Method 2: Build from Source (Advanced)

```bash
# Install build dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    cppcheck \
    doxygen \
    git \
    libaom-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavformat-dev \
    libavutil-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libgazebo11-dev \
    libgstreamer1.0-dev \
    libignition-cmake2-dev \
    libignition-common-dev \
    libignition-fuel-tools-dev \
    libignition-gui-dev \
    libignition-launch-dev \
    libignition-math6-dev \
    libignition-msgs-dev \
    libignition-physics-dev \
    libignition-rendering-dev \
    libignition-sensors-dev \
    libignition-tools-dev \
    libignition-transport-dev \
    libjxr-dev \
    libogre-1.12-dev \
    libogre-2.2-dev \
    libopencv-dev \
    libpcl-dev \
    libqt5core5a \
    libqt5gui5 \
    libqt5opengl5-dev \
    libqt5widgets5 \
    libsdformat13-dev \
    libswscale-dev \
    libtinyxml2-dev \
    libxml2-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-rosdep \
    python3-setuptools \
    software-properties-common \
    wget

# Clone and build Gazebo Garden
git clone https://github.com/gazebosim/gz-sim.git
cd gz-sim
git checkout gz-sim7
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### Verification

```bash
# Check Gazebo installation
gz --version

# Test basic functionality
gz sim --headless-testing
```

## Understanding Gazebo Architecture

### Core Components

Gazebo consists of several interconnected libraries:

- **Gz-Sim**: Simulation execution engine
- **Gz-Physics**: Physics simulation backend
- **Gz-Rendering**: Graphics rendering
- **Gz-Sensors**: Sensor simulation
- **Gz-Transport**: Message passing between components

### SDF (Simulation Description Format)

SDF is XML-based format for describing simulation environments:

```xml
<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="default">
    <!-- World properties -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>
    
    <!-- Models in the world -->
    <model name="ground_plane">
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Configuring Gazebo for Physical AI

### Performance Optimization Settings

Create or edit `~/.gz/fuel/config.yaml`:

```yaml
# Fuel configuration for Gazebo
cache_path: ~/.gz/fuel
servers:
  - url: https://fuel.gazebosim.org
    version: 1

# Performance settings
settings:
  # Reduce simulation step size for accuracy
  max_step_size: 0.001
  # Real-time update rate
  real_time_update_rate: 1000
  # Gravity (Earth standard)
  gravity: [0, 0, -9.8]
```

### Graphics Configuration

For optimal rendering performance, configure your graphics settings:

```bash
# Check graphics configuration
lspci | grep -E "VGA|3D"
nvidia-smi  # If using NVIDIA

# Set environment variables for graphics
echo 'export MESA_GL_VERSION_OVERRIDE=3.3' >> ~/.bashrc
echo 'export LIBGL_ALWAYS_SOFTWARE=0' >> ~/.bashrc  # Use hardware acceleration
```

## Creating Robot Models for Gazebo

### URDF to SDF Conversion

Gazebo uses SDF, but ROS 2 typically uses URDF. Create a simple URDF robot model in `~/physical_ai_ws/src/robot_models/urdf/physical_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="physical_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <link name="wheel_right">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="base_to_wheel_left" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="-0.2 0.2 -0.1"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="base_to_wheel_right" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right"/>
    <origin xyz="-0.2 -0.2 -0.1"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find physical_robot_control)/config/physical_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <gazebo reference="base_link">
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <gazebo reference="wheel_left">
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="wheel_right">
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>
</robot>
```

### SDF World File

Create a simulation world in `~/physical_ai_ws/src/simulation_packages/worlds/physical_ai_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="physical_ai_world">
    <!-- Physics Engine -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.2 -0.4 -0.9</direction>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Sample obstacles -->
    <model name="obstacle_1">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 0.5 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 0.5 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
    </model>

    <!-- Sample ramp -->
    <model name="ramp">
      <pose>5 0 0 0 0.3 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 2 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 2 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Gazebo-ROS 2 Integration

### Installing Gazebo ROS 2 Bridge

```bash
# Install Gazebo ROS 2 packages
sudo apt install \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros2-control \
    ros-humble-gazebo-plugins \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-joint-state-broadcaster \
    ros-humble-velocity-controllers \
    ros-humble-effort-controllers \
    ros-humble-xacro
```

### Creating Controller Configuration

Create `~/physical_ai_ws/src/control_packages/config/physical_robot_controllers.yaml`:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

velocity_controller:
  ros__parameters:
    joints:
      - base_to_wheel_left
      - base_to_wheel_right
```

### Launch File for Gazebo Integration

Create `~/physical_ai_ws/src/simulation_packages/launch/gazebo_simulation.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_world = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(
            get_package_share_directory('simulation_packages'),
            'worlds',
            'physical_ai_world.sdf'
        ),
        description='SDF world file'
    )
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': world,
            'use_sim_time': use_sim_time,
        }.items(),
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
    
    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'physical_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )
    
    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            os.path.join(
                get_package_share_directory('control_packages'),
                'config',
                'physical_robot_controllers.yaml'
            )
        ],
        output='screen'
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_world,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        controller_manager
    ])
```

## Physics Configuration for Physical AI

### Understanding Physics Parameters

Physics parameters control how objects behave in simulation:

- **Gravity**: Default is Earth's gravity (-9.8 m/s² in Z direction)
- **Friction**: Determines how objects interact with surfaces
- **Damping**: Simulates energy loss through motion
- **Stiffness**: How resistant objects are to deformation
- **Collision Detection**: How collisions are handled

### Tuning Physics for Realistic Behavior

```xml
<!-- Example physics configuration in SDF -->
<physics type="ode">
  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>
  
  <!-- Solver parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    
    <!-- Constraints -->
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_surface_layer>0.001</contact_surface_layer>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
    </constraints>
  </ode>
</physics>
```

For humanoid robots, consider these parameters:

- **Higher friction** for feet to prevent slipping
- **Proper mass distribution** for balance
- **Appropriate damping** to avoid oscillations
- **Realistic joint limits** to prevent damage

## Sensor Simulation in Gazebo

### Camera Simulation

Add camera sensors to your robot model:

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
</joint>

<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>camera/image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Simulation

Add LiDAR sensors:

```xml
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.2"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
  </inertial>
</link>

<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0.2 0 0.2" rpy="0 0 0"/>
</joint>

<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/physical_robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

## Performance Optimization

### Simulation Performance Tuning

To improve simulation performance:

1. **Reduce Physics Update Rate**: For less demanding applications
2. **Simplify Collision Models**: Use simpler shapes where possible
3. **Adjust Visual Quality**: Lower quality for headless simulations
4. **Limit Sensor Frequency**: Reduce sensor update rates when possible

### Command Line Performance Options

```bash
# Run Gazebo in headless mode (no GUI)
gz sim -s HeadlessSystem <world_file>

# Limit physics update rate
gz sim -r 100 <world_file>  # 100 Hz physics update

# Enable multi-threading
gz sim --iterations 1000 <world_file>
```

## Debugging and Troubleshooting

### Common Simulation Issues

#### Issue 1: Robot Falls Through Ground

**Problem**: Robot falls through the ground plane
**Solution**: Check collision properties and mass/inertia:

```xml
<link name="base_link">
  <!-- Must have mass and inertia -->
  <inertial>
    <mass value="10"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>
  <!-- Must have collision geometry -->
  <collision name="collision">
    <geometry>
      <box size="0.5 0.3 0.2"/>
    </geometry>
  </collision>
</link>
```

#### Issue 2: Joint Jitters or Oscillates

**Problem**: Robot joints vibrate or oscillate
**Solution**: Adjust physics parameters:

```xml
<gazebo reference="joint_name">
  <physics>
    <ode>
      <limit>
        <cfm>0.00001</cfm>  <!-- Constraint Force Mixing -->
        <erp>0.2</erp>      <!-- Error Reduction Parameter -->
      </limit>
      <spring_reference>0</spring_reference>
      <spring_stiffness>0</spring_stiffness>
    </ode>
  </physics>
</gazebo>
```

#### Issue 3: High CPU Usage

**Problem**: Simulation consumes too much CPU
**Solution**: Optimize parameters:

```bash
# Reduce physics update rate
export GZ_PHYSICS_REAL_TIME_UPDATE_RATE=500

# Use simpler collision shapes
# Limit number of active sensors
# Reduce graphics quality for headless operation
```

### Debugging Tools

#### Using Gazebo GUI for Debugging

```bash
# Launch with GUI to visualize physics properties
gz sim -g <world_file>

# In the GUI, enable:
# - Contact visualization
# - Inertia visualization
# - Joint visualization
```

#### Command Line Debugging

```bash
# Monitor simulation performance
gz topic -t /stats -n 10

# Check topic connections
ros2 topic list
ros2 topic echo /topic_name

# Monitor physics properties
gz topic -e /world/<world_name>/model/<model_name>/odometry
```

## Advanced Simulation Techniques

### Creating Complex Environments

Use Gazebo's Fuel repository for pre-built environments:

```bash
# Browse available models
gz fuel login
gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/Construction%20Site

# Use downloaded models in your worlds
```

### Custom Sensor Plugins

Create custom sensor plugins for specialized Physical AI applications:

```cpp
// Example sensor plugin header: physical_ai_sensor.hh
#include <gazebo/common/Plugin.hh>
#include <gazebo/sensors/Sensor.hh>
#include <gazebo/transport/transport.hh>

class PhysicalAISensor : public gazebo::SensorPlugin
{
public:
  void Load(gazebo::sensors::SensorPtr _sensor, sdf::ElementPtr _sdf);
  void OnUpdate();

private:
  gazebo::sensors::SensorPtr sensor;
  gazebo::transport::NodePtr node;
  gazebo::transport::PublisherPtr pub;
  std::string topicName;
};
```

### Physics Plugin Development

For custom physics behaviors in Physical AI applications:

```cpp
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>

class PhysicalAIPlugin : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr _world, sdf::ElementPtr _sdf);
  void OnUpdate(const gazebo::common::UpdateInfo &_info);

private:
  gazebo::physics::WorldPtr world;
  gazebo::event::ConnectionPtr updateConnection;
};
```

## Testing Physical AI Algorithms

### Simulation-Based Testing

Create systematic tests for your Physical AI algorithms:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time

class SimulationTester(Node):
    def __init__(self):
        super().__init__('simulation_tester')
        
        # Publishers for commands
        self.cmd_vel_pub = self.create_publisher(
            Twist, 
            '/physical_robot/cmd_vel', 
            10
        )
        
        # Subscribers for feedback
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/physical_robot/scan',
            self.scan_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/physical_robot/odom',
            self.odom_callback,
            10
        )
        
        # Test parameters
        self.test_start_time = time.time()
        self.obstacle_detected = False
        
        # Start test sequence
        self.test_timer = self.create_timer(0.1, self.run_test)
    
    def scan_callback(self, msg):
        # Check for obstacles in front
        front_scan = msg.ranges[len(msg.ranges)//2 - 90:len(msg.ranges)//2 + 90]
        if front_scan and min(front_scan) < 0.5:  # Obstacle within 50cm
            self.obstacle_detected = True
    
    def odom_callback(self, msg):
        # Track position for navigation testing
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
    
    def run_test(self):
        # Example test: Move forward until obstacle detected
        cmd = Twist()
        if not self.obstacle_detected:
            cmd.linear.x = 0.2  # Move forward
            cmd.angular.z = 0.0
            self.get_logger().info('Moving forward...')
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info('Obstacle detected, stopping.')
            
            # End test
            if time.time() - self.test_start_time > 10.0:
                self.get_logger().info('Test completed successfully')
                exit(0)
        
        self.cmd_vel_pub.publish(cmd)

def main():
    rclpy.init()
    tester = SimulationTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Takeaways

- Gazebo is essential for safe and repeatable Physical AI development
- Proper physics configuration is critical for realistic simulation
- Sensor simulation enables testing of perception algorithms
- Performance optimization is necessary for complex scenarios
- Integration with ROS 2 allows seamless development workflows
- Systematic testing in simulation improves real-world performance

## Further Reading

- "Gazebo User Guide": Complete documentation for Gazebo
- "Physics Simulation in Robotics": Advanced physics configuration
- "Gazebo ROS Integration": Detailed integration techniques
- "Simulation-Based Testing for Robotics": Best practices for testing

## Next Steps

Continue to Appendix B, Section 4: Isaac Sim Setup to configure the NVIDIA Isaac simulation environment for advanced Physical AI development.