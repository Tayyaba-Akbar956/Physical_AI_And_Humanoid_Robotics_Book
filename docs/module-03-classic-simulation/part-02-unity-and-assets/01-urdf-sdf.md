---
sidebar_position: 3
title: URDF to SDF
---

# URDF to SDF

This chapter explores the relationship between URDF (Unified Robot Description Format) and SDF (Simulation Description Format), focusing on how robot models described in URDF are converted and extended for simulation in environments like Gazebo.

## Learning Objectives

- Understand the differences between URDF and SDF formats
- Explain when and why URDF models need to be converted to SDF
- Identify the additional elements required for simulation
- Apply simulation-specific extensions to robot models
- Prepare robot models for physics simulation in Gazebo

## Simulation Environment Setup

Before diving into the details of URDF to SDF conversion, you'll need to set up your simulation environments. This chapter covers both Gazebo and Unity as potential simulation platforms.

### Gazebo Setup

Gazebo is the most common simulation environment for ROS-based robotics development. It provides physics simulation, sensor simulation, and realistic visual rendering.

#### Installing Gazebo with ROS 2

If you're using Ubuntu with ROS 2, install Gazebo as follows:

1. **Install ROS 2 with Gazebo support**:
   ```bash
   # Update package index
   sudo apt update

   # Install Gazebo packages for ROS 2
   sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

   # Install Gazebo classic (if using Gazebo Classic)
   sudo apt install ros-humble-gazebo-classic-ros
   ```

2. **Verify Installation**:
   ```bash
   # Test launching Gazebo
   ros2 run gazebo_ros gazebo

   # Check Gazebo version
   gazebo --version
   ```

3. **Install Additional Gazebo Models**:
   ```bash
   sudo apt install gazebo11-common
   # Or for newer versions:
   sudo apt install gz-sim7-common
   ```

4. **Set up environment variables** (if needed):
   ```bash
   # Add to your ~/.bashrc file
   echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
   ```

#### Gazebo Resources and Configuration

1. **Gazebo World Models**: Download additional world models from Gazebo's database
2. **Model Repository**: Set up GAZEBO_MODEL_PATH to point to your custom models
3. **Plugin Development**: Install development packages if creating custom plugins

### Unity Setup for Robotics Simulation

Unity is a powerful 3D development platform that can be used for robotics simulation, particularly for creating photorealistic environments and human-robot interaction scenarios.

#### Installing Unity for Robotics

1. **Download Unity Hub**:
   - Go to https://unity3d.com/get-unity/download
   - Download Unity Hub (recommended installation method)
   - Install Unity Hub and create an account if needed

2. **Install Unity Editor**:
   - Through Unity Hub, install Unity Editor (recommended version: 2022.3 LTS or newer)
   - When installing, select the "Linux Build Support" module if you're using ROS on Linux

3. **Install Unity Robotics Packages**:
   - Open Unity Hub and create a new 3D project
   - Go to Window → Package Manager
   - Install the following packages:
     - Unity Robotics Hub
     - ROS-TCP-Connector
     - URDF-Importer

4. **Set up ROS Communication**:
   - Install the ROS TCP Connector package in Unity
   - Install the corresponding Python package for ROS:
     ```bash
     pip install unity-robots
     ```

#### Unity Robotics Toolkit Setup

1. **Create Unity Project**:
   - Open Unity Hub and create a new 3D project
   - Name your project (e.g., "RoboticsSimulation")

2. **Import Robotics Packages**:
   - In the Package Manager, search for and install:
     - Unity Simulation for AWS RoboMaker (optional)
     - Unity Perception package (for synthetic data generation)

3. **Configure Project Settings**:
   - Go to Edit → Project Settings
   - Configure physics settings (gravity, solver iterations, etc.)
   - Set up layers and tags for different robot components

#### Setting up Unity-ROS Bridge

1. **Install the Unity ROS Bridge**:
   ```bash
   # Create a colcon workspace for ROS bridge packages
   mkdir -p ~/unity_ros_workspace/src
   cd ~/unity_ros_workspace/src

   # Clone the Unity ROS Bridge packages
   git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
   git clone https://github.com/Unity-Technologies/Unity-Robotics-Helpers.git
   ```

2. **Build the Workspace**:
   ```bash
   cd ~/unity_ros_workspace
   colcon build
   source install/setup.bash
   ```

3. **Configure Network Settings**:
   - Ensure Unity and ROS systems can communicate
   - Configure firewall settings if needed
   - Set ROS_IP environment variable appropriately

### Environment Comparison

| Aspect | Gazebo | Unity |
|--------|--------|-------|
| Primary Use | Physics-based simulation | Visual-rich simulation |
| Physics Engine | Custom physics (ODE, Bullet, DART) | PhysX |
| ROS Integration | Native via gazebo_ros packages | Via ROS TCP connector |
| Ease of Setup | Moderate | Complex |
| Learning Curve | Moderate | Steeper |
| Visual Fidelity | Good for robotics | Excellent |
| Performance | Optimized for real-time physics | Can be resource-intensive |

Choose Gazebo for traditional robotics simulation with accurate physics, or Unity for scenarios requiring high-fidelity visuals and human-robot interaction.

## Introduction: Bridging Description and Simulation

URDF and SDF serve complementary purposes in robotics development. URDF describes the robot's structure for kinematic and visual purposes, while SDF extends this description for simulation-specific needs. The relationship between these formats is crucial for bringing robot models into simulation environments.

URDF was developed primarily for ROS and focuses on:
- Kinematic structure (links connected by joints)
- Visual representation for RViz and other tools
- Basic inertial properties for control

SDF was developed for simulation and includes:
- Simulation-specific properties (physics, sensors, controllers)
- World description (environment, lighting, physics engine)
- Gazebo-specific extensions (plugins, visualization)

### The Conversion Process

In practice, URDF models are often automatically converted to SDF when loaded into simulation, but understanding the process helps in creating better simulation-ready models.

## Core Concepts

### Format Differences

**URDF** is XML-based with a focus on the robot's kinematic chain:
```xml
<robot name="my_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.5" radius="0.2"/>
      </geometry>
    </visual>
    <inertial>
      <mass value="10"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

**SDF** is more comprehensive for simulation:
```xml
<sdf version="1.7">
  <model name="my_robot">
    <pose>0 0 0 0 0 0</pose>
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>10</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <cylinder>
            <length>0.5</length>
            <radius>0.2</radius>
          </cylinder>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <cylinder>
            <length>0.5</length>
            <radius>0.2</radius>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
```

### Key Translation Elements

When translating from URDF to SDF, certain elements undergo changes:

**Inertial Specifications**:
- URDF uses `ixx="0.1"` attributes
- SDF uses nested `<ixx>0.1</ixx>` elements

**Geometry Representations**:
- URDF uses `<box size="0.5 0.2 0.1"/>`
- SDF uses `<box><size>0.5 0.2 0.1</size></box>`

**Coordinate Frames**:
- URDF has implicit base frame
- SDF uses explicit `<pose>` elements

### Gazebo Extensions

URDF can be extended with Gazebo-specific elements that are converted to SDF:

```xml
<!-- In URDF, Gazebo extensions appear as <gazebo> tags -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <self_collide>true</self_collide>
</gazebo>

<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>robot1</namespace>
    </ros>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.1</wheel_diameter>
  </plugin>
</gazebo>
```

## Practical Implementation

### Automatic URDF to SDF Conversion

Gazebo can automatically convert URDF to SDF when spawning models:

```python
# Using gazebo_ros spawn_entity
import subprocess

# Spawn URDF model in Gazebo
subprocess.run([
    'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
    '-entity', 'my_robot',
    '-topic', 'robot_description',
    '-x', '0', '-y', '0', '-z', '0.5'
])
```

### Required Elements for Simulation

While URDF is sufficient for visualization, simulation requires additional elements:

**Collision Models**: URDF typically includes only visual geometry, but simulation needs collision geometry:

```xml
<!-- In URDF, we add explicit collision geometry -->
<link name="base_link">
  <visual>
    <geometry>
      <mesh filename="package://my_robot/meshes/base_visual.stl"/>
    </geometry>
  </visual>
  
  <!-- Add collision geometry -->
  <collision>
    <geometry>
      <mesh filename="package://my_robot/meshes/base_collision.stl"/>
      <!-- Or use simplified geometry for performance -->
      <!-- <box size="0.5 0.3 0.2"/> -->
    </geometry>
  </collision>
  
  <inertial>
    <mass value="10"/>
    <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
  </inertial>
</link>
```

### Gazebo-Specific Extensions

To make URDF work effectively in simulation, add Gazebo-specific extensions:

```xml
<!-- Gazebo extensions in URDF -->
<gazebo reference="base_link">
  <!-- Visual properties in Gazebo -->
  <material>
    <ambient>0.2 0.2 0.8 1.0</ambient>
    <diffuse>0.4 0.4 1.0 1.0</diffuse>
    <specular>0.1 0.1 0.1 1.0</specular>
  </material>
  
  <!-- Physics properties -->
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<!-- Sensor plugins -->
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <always_on>true</always_on>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <plugin name="ray_sensor" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>robot1</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>

<!-- Controller plugins -->
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>robot1</namespace>
      <remapping>cmd_vel:=cmd_vel</remapping>
      <remapping>odom:=odom</remapping>
    </ros>
    <update_rate>30</update_rate>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.1</wheel_diameter>
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
    <publish_wheel_tf>true</publish_wheel_tf>
  </plugin>
</gazebo>
```

### Complete Example: Converting a Simple Robot

Here's how to prepare a simple two-wheeled robot model for Gazebo simulation:

**Original URDF**:
```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>

  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel"/>
    <origin xyz="0 0.2 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

**Extended URDF for Gazebo**:
```xml
<?xml version="1.0"?>
<robot name="simple_robot_gazebo">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>

  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <!-- Add collision geometry -->
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <!-- Add collision geometry -->
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel"/>
    <origin xyz="0 0.2 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Gazebo-specific extensions -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="wheel">
    <material>Gazebo/Blue</material>
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
  </gazebo>

  <!-- Differential drive controller -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>robot1</namespace>
      </ros>
      <left_joint>wheel_joint</left_joint>  <!-- Note: This is a simplified example -->
      <right_joint>wheel_joint</right_joint> <!-- In reality, you'd have two wheels -->
      <wheel_separation>0.4</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>
</robot>
```

### Using xacro for Complex Conversions

Xacro can help manage the complexity of simulation-ready URDF files:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot_gazebo">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Macro for creating wheels with simulation properties -->
  <xacro:macro name="wheel" params="name parent xyz radius wheel_mass mu friction">
    <joint name="${name}_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <origin xyz="${xyz}" rpy="${M_PI/2} 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>

    <link name="${name}_link">
      <visual>
        <geometry>
          <cylinder length="0.05" radius="${radius}"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.05" radius="${radius}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${wheel_mass}"/>
        <inertia
          ixx="${wheel_mass * radius * radius / 2}"
          ixy="0" ixz="0"
          iyy="${wheel_mass * (3 * radius * radius + 0.05 * 0.05) / 12}"
          iyz="0"
          izz="${wheel_mass * radius * radius / 2}" />
      </inertial>
    </link>

    <!-- Simulation properties for this wheel -->
    <gazebo reference="${name}_link">
      <material>Gazebo/Blue</material>
      <mu1>${mu}</mu1>
      <mu2>${mu}</mu2>
      <fdir1>1 0 0</fdir1>
    </gazebo>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:wheel name="left_wheel" parent="base_link" xyz="0 0.2 0" radius="0.1" wheel_mass="0.2" mu="1.0" friction="0.01" />
  <xacro:wheel name="right_wheel" parent="base_link" xyz="0 -0.2 0" radius="0.1" wheel_mass="0.2" mu="1.0" friction="0.01" />

  <!-- Add more elements as needed -->
</robot>
```

## Code Examples for Creating Simulation Worlds and Robot Models

### Example 1: Simple Gazebo World with Robot

**Creating a Gazebo World File** (`simple_room.world`):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a sun light source -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define a simple room -->
    <model name="room">
      <!-- Floor -->
      <link name="floor">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <pose>0 0 -0.05 0 0 0</pose>
      </link>

      <!-- Walls -->
      <link name="wall_1">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <pose>0 5 1.5 0 0 0</pose>
      </link>

      <link name="wall_2">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <pose>0 -5 1.5 0 0 0</pose>
      </link>

      <link name="wall_3">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 10 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 10 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <pose>5 0 1.5 0 0 0</pose>
      </link>

      <link name="wall_4">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 10 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 10 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <pose>-5 0 1.5 0 0 0</pose>
      </link>
    </model>

    <!-- Place your robot in the world -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Example 2: Python Script to Spawn Robot in Gazebo

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.srv import SpawnEntity
import time

class RobotSpawner(Node):
    def __init__(self):
        super().__init__('robot_spawner')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Robot description (in real use, this would come from a file or parameter)
        self.robot_description = """
        <?xml version="1.0"?>
        <robot name="simple_robot">
          <link name="base_link">
            <visual>
              <geometry>
                <cylinder length="0.1" radius="0.2"/>
              </geometry>
              <material name="blue">
                <color rgba="0 0 0.8 1"/>
              </material>
            </visual>
            <collision>
              <geometry>
                <cylinder length="0.1" radius="0.2"/>
              </geometry>
            </collision>
            <inertial>
              <mass value="5"/>
              <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
            </inertial>
          </link>
        </robot>
        """

    def spawn_robot(self):
        req = SpawnEntity.Request()
        req.name = "my_robot"
        req.xml = self.robot_description
        req.robot_namespace = ""
        req.initial_pose.position.x = 0.0
        req.initial_pose.position.y = 0.0
        req.initial_pose.position.z = 0.5
        req.initial_pose.orientation.x = 0.0
        req.initial_pose.orientation.y = 0.0
        req.initial_pose.orientation.z = 0.0
        req.initial_pose.orientation.w = 1.0

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f'Successfully spawned robot: {future.result().status_message}')
        else:
            self.get_logger().error('Failed to spawn robot')

def main():
    rclpy.init()
    spawner = RobotSpawner()

    # Allow Gazebo to start up
    time.sleep(5)

    spawner.spawn_robot()

    # Keep the node alive to handle potential callbacks
    rclpy.spin(spawner)

if __name__ == '__main__':
    main()
```

### Example 3: Launch File for Simulation Environment

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
import yaml

def generate_launch_description():
    # World file path
    world_file = PathJoinSubstitution([
        FindPackageShare('my_robot_gazebo'),
        'worlds',
        'simple_room.world'
    ])

    # Launch Gazebo with the world
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'false',
        }.items()
    )

    # Spawn the robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0',
            '-y', '0',
            '-z', '0.5'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': """
                <?xml version="1.0"?>
                <robot name="simple_robot">
                  <link name="base_link">
                    <visual>
                      <geometry>
                        <cylinder length="0.1" radius="0.2"/>
                      </geometry>
                      <material name="blue">
                        <color rgba="0 0 0.8 1"/>
                      </material>
                    </visual>
                    <collision>
                      <geometry>
                        <cylinder length="0.1" radius="0.2"/>
                      </geometry>
                    </collision>
                    <inertial>
                      <mass value="5"/>
                      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
                    </inertial>
                  </link>
                </robot>
            """
        }]
    )

    return LaunchDescription([
        gazebo_launch,
        robot_state_publisher,
        spawn_entity,
    ])
```

### Example 4: Creating a Simpler Robot Model for Initial Testing

```xml
<?xml version="1.0"?>
<robot name="test_bot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.06"/>
    </inertial>
  </link>

  <!-- Simple caster wheel for balance -->
  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Fixed joint connecting caster to base -->
  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="0.2 0 -0.15" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo-specific properties -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="caster_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.01</mu1>
    <mu2>0.01</mu2>
  </gazebo>
</robot>
```

## Troubleshooting Common Issues

### Collision vs. Visual Mismatch

Sometimes collision and visual geometry have different origins:

```xml
<!-- Visual and collision in different positions -->
<link name="sensor_link">
  <visual>
    <!-- Visual centered at link origin -->
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.1" radius="0.05"/>
    </geometry>
  </visual>
  <collision>
    <!-- Collision offset from link origin -->
    <origin xyz="0.02 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.08 0.08 0.1"/>
    </geometry>
  </collision>
</link>
```

### Inertial Calculation Errors

Incorrect inertial properties can cause simulation instability:

```xml
<!-- Example of calculating inertial properties for a cylinder -->
<!--
For a solid cylinder:
- mass = density * volume = density * π * r² * h
- ixx = iyy = (1/12) * m * (3*r² + h²)
- izz = (1/2) * m * r²
-->

<link name="cylinder_link">
  <inertial>
    <mass value="1.0"/>
    <inertia
      ixx="0.009166" ixy="0" ixz="0"  <!-- 1/12 * 1.0 * (3*0.1² + 0.2²) = 1/12 * (0.03 + 0.04) -->
      iyy="0.009166" iyz="0"
      izz="0.005" /> <!-- 1/2 * 1.0 * 0.1² -->
  </inertial>
</link>
```

## Best Practices

### Keep URDF Clean

- Maintain a clean URDF for general use
- Add Gazebo-specific elements in extensions
- Use xacro to manage complexity
- Document simulation-specific parameters

### Optimize for Performance

- Use simplified collision geometries when possible
- Keep inertial parameters realistic but not overly complex
- Use appropriate mesh resolution for visual and collision elements

### Validate Your Model

- Test the model in simulation before complex tasks
- Check for joint limits and singularities
- Verify physical properties match real robot where possible

## Practical Exercises for Creating and Testing Simulation Environments

### Exercise 1: Basic Robot Simulation Setup
**Objective**: Create a complete simulation environment with a simple robot.

1. Create a URDF file for a simple robot (e.g., a box-shaped base with wheels)
2. Add Gazebo extensions for physics simulation
3. Create a world file with basic obstacles
4. Create a launch file that starts Gazebo with your world and spawns your robot
5. Launch the simulation and verify that the robot appears in the environment

**Steps to Complete**:
```
# 1. Create robot URDF with Gazebo extensions
# 2. Create a world file in worlds/ directory
# 3. Create a launch file to bring up the simulation
# 4. Build and run: ros2 launch your_package simulation.launch.py
```

### Exercise 2: Sensor Integration in Simulation
**Objective**: Add sensors to your robot and verify they publish data correctly.

1. Add a LIDAR sensor to your robot model using Gazebo plugins
2. Add a camera sensor to your robot
3. Launch the simulation and verify that sensor data is published on ROS topics
4. Use rviz2 to visualize the sensor data
5. Test that the sensor outputs change when the robot or environment changes

**Verification Steps**:
```
# 1. Check topics: ros2 topic list | grep scan
# 2. View laser scan: ros2 topic echo /laser_scan
# 3. View camera: ros2 run image_view image_view
# 4. Visualize in rviz2
```

### Exercise 3: Physics Properties Tuning
**Objective**: Understand how physics properties affect robot behavior.

1. Create a robot model with basic geometry and inertial properties
2. Launch it in simulation and observe its behavior
3. Adjust friction coefficients and observe the differences
4. Change inertial properties and test how it affects movement
5. Modify contact properties and see how it impacts collisions

**Parameters to Experiment With**:
- `<mu1>` and `<mu2>`: Primary and secondary friction coefficients
- `<kp>` and `<kd>`: Contact stiffness and damping parameters
- Inertial properties: mass, center of mass, moments of inertia
- Joint limits and friction

### Exercise 4: Environment Design and Testing
**Objective**: Create a custom environment for robot testing.

1. Design a simple world with obstacles (walls, boxes, ramps)
2. Add different surface properties (high/low friction)
3. Create different lighting conditions
4. Test how your robot navigates the environment
5. Identify and fix any issues with collisions or physics

**World Design Tasks**:
- Create a maze-like structure for navigation testing
- Add objects of different masses and frictions
- Design a space with narrow passages
- Include elevated surfaces or ramps to test climbing

### Exercise 5: Controller Integration
**Objective**: Connect a ROS 2 controller to your simulated robot.

1. Build upon your robot model to include ros2_control elements
2. Create a hardware interface for simulation
3. Implement a basic controller (e.g., joint_trajectory_controller)
4. Test the controller in simulation
5. Verify that commands sent to the controller affect the robot in simulation

**Implementation Steps**:
```xml
<!-- Add ros2_control interface to your URDF -->
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
  <!-- Define joints and transmissions -->
</ros2_control>
```

### Exercise 6: Safety and Collision Testing
**Objective**: Test robot behavior in potential collision scenarios.

1. Design test scenarios where collisions might occur
2. Adjust safety parameters to prevent instabilities
3. Test how the robot behaves when hitting obstacles
4. Implement collision detection algorithms
5. Create a safety system that prevents crashes

**Testing Scenarios**:
- Robot approaching a wall at different speeds
- Robot navigating through narrow spaces
- Robot carrying objects and avoiding collisions
- Multiple robots operating in the same space

### Exercise 7: Performance Optimization
**Objective**: Optimize your simulation for better performance.

1. Create models with different levels of detail (high and low poly)
2. Test simulation performance with each model
3. Compare simulation with box approximation vs. mesh collision geometry
4. Identify bottlenecks in your simulation setup
5. Document the trade-offs between realism and performance

**Optimization Techniques**:
- Use simplified collision geometry
- Reduce mesh complexity
- Adjust physics engine parameters
- Optimize controller update rates
- Limit sensor update frequencies appropriately

### Exercise 8: Integration Testing
**Objective**: Test the complete system with multiple components.

1. Combine your robot with a pre-built world
2. Integrate sensors, controllers, and navigation stack
3. Perform a complete simulation run
4. Debug any issues that arise
5. Document the complete simulation setup

**Final Testing Tasks**:
- Test robot navigation from point A to B
- Verify sensor data consistency
- Check that robot doesn't exhibit unstable behavior
- Evaluate overall performance and stability

## Hands-on Exercise

1. **Conversion Exercise**: Take the URDF robot model you created in Module 2 and extend it with Gazebo-specific elements for simulation.

2. **Collision Geometry**: Add appropriate collision geometry to a robot model, comparing the effects of different collision shapes (box vs. cylinder vs. mesh).

3. **Inertial Validation**: Calculate and verify inertial properties for different geometries using physics formulas.

4. **Plugin Integration**: Add a sensor plugin (e.g., camera, LIDAR) to a robot model and verify it works in simulation.

5. **Performance Optimization**: Compare simulation performance with different levels of geometric complexity.

## Key Takeaways

- URDF describes robot kinematics and visuals, SDF extends to simulation
- Gazebo-specific extensions allow URDF to be used in simulation
- Collision geometry is essential for physics simulation
- Inertial properties must be accurate for realistic simulation
- Xacro can manage complexity in simulation-ready URDF files
- Proper conversion is essential for successful robot simulation

## Further Reading

- Gazebo Model Format Documentation
- URDF to SDF Conversion Guide
- "Simulation of Robotic Systems" by Dudek and Jenkin
- Xacro Best Practices

## Next Steps

Continue to Chapter 2: Unity Introduction to explore Unity as an alternative simulation environment.