---
sidebar_position: 5
title: Launch Files
---

# Launch Files

This chapter covers ROS 2 launch files, which provide a way to configure and start multiple nodes and system components simultaneously. Launch files are essential for deploying complex robotic systems with multiple coordinated components.

## Learning Objectives

- Understand the purpose and format of ROS 2 launch files
- Create launch files for single and multiple node systems
- Implement launch file parameters and conditional execution
- Use launch file includes and composition patterns
- Apply launch files to real-world robot deployment scenarios

## Introduction: Orchestrating Robot Systems

A complete robotic system typically consists of many coordinated software components:
- Perception nodes (processing sensor data)
- Planning nodes (determining robot actions)
- Control nodes (executing movements)
- Visualization nodes (monitoring system state)
- Hardware interface nodes (communicating with physical devices)

Starting these components manually would be time-consuming and error-prone. Launch files provide a way to start multiple nodes with appropriate configurations simultaneously, enabling reproducible and consistent system deployment.

## Core Concepts

### Launch Files vs. Command Line

Without launch files, starting a robot system might require multiple command-line commands:

```bash
# Manual startup sequence
ros2 run sensor_package camera_driver_node
ros2 run sensor_package lidar_driver_node
ros2 run navigation_package costmap_node
ros2 run navigation_package planner_node
ros2 run controller_package joint_state_controller
```

With launch files, all components can be started with a single command:
```bash
ros2 launch my_robot_package robot.launch.py
```

### Launch File Formats

ROS 2 supports launch files in multiple formats:

1. **Python launch files** (.py): Most common, allows full Python scripting
2. **XML launch files** (.launch): Declarative format
3. **YAML launch files** (.yaml): Configuration-focused format

Python launch files are most commonly used as they provide maximum flexibility.

### Launch File Components

Launch files contain several key elements:

- **Nodes**: Define which ROS nodes to start with their configurations
- **Parameters**: Set node parameters at startup
- **Remappings**: Redirect topic/service names
- **Conditions**: Control execution based on parameters or environment
- **Includes**: Include other launch files for modularity
- **Actions**: General executable actions beyond nodes

## Practical Implementation

### Basic Python Launch File

Here's a simple Python launch file that starts a single node:

```python
# basic_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim',
            output='screen'
        )
    ])
```

### Launch File with Multiple Nodes

Here's a launch file that starts multiple nodes with configurations:

```python
# multi_node_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='robot1',
        description='Robot namespace'
    )
    
    # Get launch configuration
    namespace = LaunchConfiguration('namespace')
    
    return LaunchDescription([
        namespace_arg,
        
        # Launch the robot simulator
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim',
            namespace=namespace,
            parameters=[
                {'background_r': 69},
                {'background_g': 86},
                {'background_b': 98}
            ],
            output='screen'
        ),
        
        # Launch the controller
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop',
            namespace=namespace,
            remappings=[
                ('/turtle1/cmd_vel', ['/', namespace, '/cmd_vel'])
            ],
            output='screen',
            # Only run if a condition is met
            condition=None  # In real usage, you'd use a condition here
        )
    ])
```

### Launch File with Parameters

Here's a launch file that accepts parameters and uses them to configure nodes:

```python
# parameterized_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os

def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            os.path.dirname(__file__),
            'config',
            'robot_config.yaml'
        ),
        description='Path to configuration file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # Get launch configurations
    config_file = LaunchConfiguration('config_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    return LaunchDescription([
        config_file_arg,
        use_sim_time_arg,
        
        # Robot state publisher with parameters
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                config_file,
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),
        
        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        )
    ])
```

### Launch File with Conditional Execution

Conditional execution allows launching different components based on parameters:

```python
# conditional_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    # Declare launch arguments
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # Get launch configurations
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    return LaunchDescription([
        use_rviz_arg,
        use_sim_time_arg,
        
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),
        
        # Launch RViz conditionally
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/path/to/config.rviz'],
            output='screen',
            condition=IfCondition(use_rviz)  # Only launch if use_rviz is true
        ),
        
        # Other node that should NOT run with sim time
        Node(
            package='diagnostic_aggregator',
            executable='aggregator_node',
            name='diagnostic_aggregator',
            output='screen',
            condition=UnlessCondition(use_sim_time)  # Only launch if use_sim_time is false
        )
    ])
```

### Launch File Includes

Modular launch files can include other launch files:

```python
# modular_launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Include sensor drivers launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('my_sensor_package'),
                    'launch',
                    'sensor_drivers.launch.py'
                ])
            ])
        ),
        
        # Include navigation launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('my_navigation_package'),
                    'launch',
                    'navigation.launch.py'
                ])
            ])
        ),
        
        # Include robot controller
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('my_controller_package'),
                    'launch',
                    'controller.launch.py'
                ])
            ])
        )
    ])
```

### Complex Robot Launch Example

Here's a comprehensive launch file for a mobile robot:

```python
# mobile_robot_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    robot_namespace = LaunchConfiguration('robot_namespace')
    params_file = LaunchConfiguration('params_file')
    
    # Launch arguments
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    declare_use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    )
    
    declare_robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value='',
        description='Robot namespace'
    )
    
    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_package'),
            'config',
            'navigation_params.yaml'
        ]),
        description='Full path to params file for all nodes'
    )
    
    # Nodes
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=robot_namespace,
        parameters=[ParameterFile(param_file=params_file, allow_substitutions=True)],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ]
    )
    
    start_lifecycle_manager_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        namespace=robot_namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server',
                                   'amcl',
                                   'bt_navigator',
                                   'controller_server',
                                   'planner_server',
                                   'recoveries_server',
                                   'velocity_smoother',
                                   'pose_smoother']}]
    )
    
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('my_robot_package'),
            'rviz',
            'default.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz)
    )
    
    # Create the launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_arg)
    ld.add_action(declare_use_rviz_arg)
    ld.add_action(declare_robot_namespace_arg)
    ld.add_action(declare_params_file_arg)
    
    # Add nodes
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_lifecycle_manager_cmd)
    ld.add_action(start_rviz_cmd)
    
    return ld
```

### Launch File with Custom Actions

Launch files can perform actions beyond starting nodes:

```python
# action_launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution

def generate_launch_description():
    return LaunchDescription([
        # Log a message
        LogInfo(msg=['Starting robot system...']),
        
        # Execute an external process
        ExecuteProcess(
            cmd=['ros2', 'param', 'set', '/robot_state_publisher', 'use_sim_time', 'true'],
            output='screen'
        ),
        
        # Delayed node startup (after 3 seconds)
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='turtlesim',
                    executable='turtlesim_node',
                    name='sim'
                )
            ]
        ),
        
        # Execute a shell script after nodes start
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=['/path/to/setup_script.sh'],
                    output='screen'
                )
            ]
        )
    ])
```

## Best Practices

### Organization and Modularity

1. **Separate concerns**: Create different launch files for different subsystems
2. **Use parameters**: Make launch files configurable through parameters rather than hardcoding values
3. **Include common components**: Use IncludeLaunchDescription to compose complex systems from simpler parts
4. **Document launch arguments**: Always document what parameters your launch files accept

### Performance Considerations

1. **Minimize startup time**: Only launch what's needed for the particular use case
2. **Use conditions**: Use IfCondition and UnlessCondition to avoid launching unnecessary nodes
3. **Group related nodes**: Launch related nodes together to simplify system configuration

## Hands-on Exercise

1. **Basic Launch Exercise**: Create a launch file that starts the turtlesim node and the keyboard teleop node together.

2. **Parameterized Launch**: Create a launch file that accepts parameters for robot name and starts nodes with that namespace.

3. **Conditional Launch**: Create a launch file that conditionally starts RViz based on a parameter.

4. **Multi-package Launch**: Create a master launch file that includes launch files from different packages to start a complete robot system.

5. **Real-world Application**: Design launch files for the robot you described in the URDF exercise, including driver nodes, state publishers, and visualization tools.

## Key Takeaways

- Launch files orchestrate the startup of multiple ROS nodes and components
- Python launch files offer maximum flexibility with full Python scripting capability
- Parameters and conditions allow launch files to be configurable for different scenarios
- IncludeLaunchDescription enables modular composition of complex systems
- Launch files are essential for reproducible robot system deployment
- Proper launch file design improves system maintainability and usability

## Further Reading

- ROS 2 Launch Documentation
- "Programming Robots with ROS" by Quigley et al.
- ROS 2 Tutorials: Launch files

## Next Steps

Continue to Module 3 to learn about simulation environments for physical AI systems.