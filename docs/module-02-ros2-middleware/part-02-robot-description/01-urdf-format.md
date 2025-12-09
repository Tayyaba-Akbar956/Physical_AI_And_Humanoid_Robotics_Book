---
sidebar_position: 4
title: URDF Format
---

# URDF Format

This chapter explores the Unified Robot Description Format (URDF), the XML-based format used to describe robot models in ROS. URDF is fundamental to robot simulation, visualization, and control in ROS-based systems.

## Learning Objectives

- Understand the purpose and structure of URDF
- Create basic URDF models for simple robots
- Define links, joints, and their properties in URDF
- Apply visual and collision properties to robot models
- Integrate URDF with ROS tools and simulation environments

## Introduction: Describing Robots in XML

The Unified Robot Description Format (URDF) is an XML-based format used to describe robots in ROS. A URDF file contains a complete description of a robot's physical and kinematic properties, including:

- The robot's kinematic chain (links connected by joints)
- Physical properties (mass, inertia, geometry)
- Visual appearance (for simulation and visualization)
- Collision properties (for physics simulation)
- Sensor and actuator locations

URDF is essential for robot simulation, visualization (RViz), and kinematic analysis tools in ROS. It provides a standardized way to describe a robot that can be used by various tools and algorithms.

## Core Concepts

### Links

A **link** in URDF represents a rigid body part of the robot:

- Each link has a name and properties (visual, collision, inertial)
- Links are the building blocks of robot models
- Each link has a coordinate frame (frame of reference)
- At least one link must be defined as the "base" or "root" link
- Links cannot move relative to themselves (they are rigid)

### Joints

A **joint** connects two links and defines their relative motion:

- **Fixed joints**: No relative motion between links
- **Revolute joints**: Single axis of rotation, limited to a range
- **Continuous joints**: Single axis of rotation, unlimited
- **Prismatic joints**: Single axis of linear motion, limited
- **Planar joints**: Motion on a plane
- **Floating joints**: 6 degrees of freedom (no constraints)

### Robot Base

Every robot has a base link that serves as the reference frame. All other links are connected to the base through a chain of joints. The base link is typically the robot's main body or the part that connects to the world.

### Materials and Colors

URDF supports material definitions that specify visual properties:

- Color (RGBA values)
- Texture (reference to image files)
- Shading properties

## Practical Implementation

### Basic URDF Structure

Here's the basic structure of a URDF file:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Define materials -->
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
  
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>

  <!-- Define the base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Define a wheel link -->
  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Connect base and wheel with a joint -->
  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 -0.3 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

### Defining Visual Elements

Visual elements define how a robot appears in simulation and visualization:

```xml
<visual>
  <!-- Position and orientation relative to the link frame -->
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  
  <!-- Geometry of the visual element -->
  <geometry>
    <!-- Box with dimensions -->
    <box size="0.5 0.2 0.1"/>
    
    <!-- Sphere with radius -->
    <!-- <sphere radius="0.1"/> -->
    
    <!-- Cylinder with radius and length -->
    <!-- <cylinder radius="0.1" length="0.2"/> -->
    
    <!-- Mesh file -->
    <!-- <mesh filename="package://my_robot/meshes/link1.stl"/> -->
  </geometry>
  
  <!-- Material applied to the geometry -->
  <material name="white"/>
</visual>
```

### Defining Collision Elements

Collision elements define the physical volume for physics simulation:

```xml
<collision>
  <!-- Position and orientation relative to the link frame -->
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  
  <!-- Geometry of the collision element -->
  <geometry>
    <box size="0.5 0.2 0.1"/>
  </geometry>
</collision>
```

### Defining Inertial Properties

Inertial properties define the mass distribution for physics simulation:

```xml
<inertial>
  <!-- Mass in kilograms -->
  <mass value="1.0"/>
  
  <!-- Inertia matrix (3x3 symmetric matrix) -->
  <inertia 
    ixx="0.1" ixy="0.0" ixz="0.0"
    iyy="0.1" iyz="0.0"
    izz="0.1" />
</inertial>
```

### Defining Joints

Joints connect links and define their relative motion:

```xml
<joint name="joint_name" type="revolute">
  <!-- Parent link (the link this joint connects FROM) -->
  <parent link="base_link"/>
  
  <!-- Child link (the link this joint connects TO) -->
  <child link="arm_link"/>
  
  <!-- Origin of the joint relative to the parent link frame -->
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
  
  <!-- Axis of rotation/translation -->
  <axis xyz="0 0 1"/>
  
  <!-- Joint limits (for revolute and prismatic joints) -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  
  <!-- Joint dynamics -->
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Complete Example: Simple Mobile Robot

Here's a complete example of a simple two-wheeled mobile robot:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Base to left wheel joint -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0.15 0.18 0" rpy="-1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Base to right wheel joint -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0.15 -0.18 0" rpy="-1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Optional: Add a caster wheel for stability -->
  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Fixed joint for caster wheel -->
  <joint name="caster_wheel_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="-0.2 0 -0.05" rpy="0 0 0"/>
  </joint>
</robot>
```

## Xacro: URDF with Macros

Xacro is a macro language that makes complex URDF files more manageable:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">

  <!-- Define properties using xacro properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_width" value="0.5" />
  <xacro:property name="base_length" value="0.3" />
  <xacro:property name="base_height" value="0.1" />

  <!-- Define a macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent x_reflect y_reflect">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${x_reflect*0.15} ${y_reflect*0.18} 0" rpy="${-M_PI/2} 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder length="0.05" radius="0.1"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.05" radius="0.1"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="left" parent="base_link" x_reflect="1" y_reflect="1"/>
  <xacro:wheel prefix="right" parent="base_link" x_reflect="1" y_reflect="-1"/>

</robot>
```

## Hands-on Exercise

1. **Basic URDF Exercise**: Create a URDF file for a simple robot with a rectangular base and two wheels. Use proper visual, collision, and inertial properties.

2. **Xacro Exercise**: Convert your simple robot URDF to use Xacro macros. Create a parameterized macro that can generate multiple similar components.

3. **Kinematic Chain Exercise**: Create a URDF for a simple 3-DOF arm (shoulder, elbow, wrist) and verify the kinematic chain is correct.

4. **Simulation Preparation**: Research how to use your URDF file in Gazebo simulation, including how to add transmission plugins for joint control.

## Key Takeaways

- URDF describes robot geometry, kinematics, and physical properties
- Links represent rigid bodies, joints define connections and relative motion
- Visual, collision, and inertial properties define different aspects of links
- Xacro simplifies complex URDF files with macros and parameters
- URDF is essential for simulation, visualization, and planning tools
- Proper inertial properties are crucial for accurate physics simulation

## Further Reading

- ROS URDF Tutorials
- "Robotics, Vision and Control" by Corke
- Xacro documentation and tutorials
- Gazebo simulation with URDF models

## Next Steps

Continue to Chapter 2: Launch Files to learn how to configure and launch complex robot systems.