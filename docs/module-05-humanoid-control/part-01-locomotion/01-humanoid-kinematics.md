---
sidebar_position: 1
title: Humanoid Kinematics
---

# Humanoid Kinematics

This chapter explores the kinematic principles underlying humanoid robots, focusing on the mathematical models that describe how these complex robots move and interact with their environment. Humanoid kinematics involve understanding the relationships between joint angles and the positions and orientations of various body parts.

## Learning Objectives

- Understand the kinematic structure of humanoid robots
- Apply forward and inverse kinematics to humanoid systems
- Identify the challenges specific to humanoid locomotion
- Implement kinematic solutions for humanoid robots
- Analyze the constraints and degrees of freedom in humanoid systems

## Introduction: The Complexity of Humanoid Motion

Humanoid robots present unique kinematic challenges compared to simpler robotic systems. With multiple degrees of freedom (DOF) distributed across legs, arms, torso, and head, these robots must solve complex kinematic problems for stable locomotion and manipulation. Unlike wheeled robots or robotic arms, humanoid systems must maintain balance while moving through 3D environments.

The human body has approximately 200+ joints, but humanoid robots typically have 20-50 actuated joints, carefully selected to enable essential movements for navigation and interaction. The kinematic structure of a humanoid robot affects its ability to walk, run, climb, manipulate objects, and maintain stability.

### Key Challenges in Humanoid Kinematics

1. **Balance Maintenance**: Humanoid robots must maintain their center of mass (CoM) within their support polygon to avoid falling
2. **Multi-Limbed Coordination**: Arms and legs must be coordinated to maintain balance while performing tasks
3. **Obstacle Avoidance**: Multiple limbs must be coordinated to avoid self-collision and environmental obstacles
4. **Dynamic Stability**: Maintaining balance during movement, not just in static poses

### Humanoid vs. Human Motion

While humanoid robots aim to replicate human-like movement, there are fundamental differences:

- **Degrees of Freedom**: Humans have redundant DOF in many joints, allowing for multiple solutions to the same task
- **Compliance**: Human joints have built-in compliance that helps with shock absorption and safety
- **Adaptability**: Human motion can adapt in real-time to unexpected perturbations
- **Energy Efficiency**: Human biomechanics have evolved for energy-efficient locomotion

## Core Concepts

### Kinematic Chains

A humanoid robot consists of multiple kinematic chains that must work together:

- **Leg Chains**: From hip to foot, enabling locomotion and balance
- **Arm Chains**: From shoulder to hand, for manipulation tasks
- **Spine Chain**: Torso movement and upper body orientation
- **Head Chain**: Neck movement for vision and interaction

### Coordinate Systems

Humanoid kinematics typically use several coordinate systems:

- **World Frame**: Fixed reference frame for the environment
- **Base Frame**: Usually located at the robot's pelvis or between the feet
- **Link Frames**: Attached to each rigid body segment
- **End-Effector Frames**: Located at hands and feet
- **Center of Mass Frame**: For balance control

### Degrees of Freedom Analysis

A typical humanoid robot has:
- 6 DOF in each leg (hip: 3 DOF, knee: 1 DOF, ankle: 2 DOF)
- 7 DOF in each arm (shoulder: 3 DOF, elbow: 1 DOF, wrist: 3 DOF)
- 3-6 DOF in the torso/spine
- 2-3 DOF in the neck/head
- Total: 28+ DOF for a basic humanoid

## Mathematical Framework

### Forward Kinematics

Forward kinematics determines the position and orientation of each link given the joint angles. For humanoid robots, this involves complex transformations through multiple kinematic chains.

For a humanoid leg, the transformation from hip to foot can be expressed as:
```
T_foot = T_hip * A1(θ1) * A2(θ2) * ... * A6(θ6)
```
Where `A_i(θ_i)` represents the transformation matrix for joint i with angle θ_i.

### Inverse Kinematics

Inverse kinematics determines the required joint angles to achieve a desired end-effector position and orientation. For humanoid robots, this is often an ill-posed problem with multiple solutions, requiring optimization criteria such as:

- Minimizing joint movement from current configuration
- Maintaining joint limits
- Avoiding singularities
- Optimizing for balance

### Balance and Locomotion

Humanoid locomotion requires special attention to:

**Zero-Moment Point (ZMP)**:
- A point where the net moment of ground reaction forces equals zero
- Critical for stable walking
- Must remain within the support polygon (foot area)

**Capture Point**:
- A point where the robot can step to stop its motion
- Important for recovery from disturbances

**Center of Mass (CoM)**:
- Must be controlled to remain within the support polygon
- Used in balance control strategies

## Practical Implementation

### Denavit-Hartenberg Parameters for Humanoid Robots

While the Denavit-Hartenberg (DH) convention is commonly used for robot arms, humanoid robots often use alternative representations due to their complex structure. However, DH parameters can still be applied to individual limbs:

For a humanoid leg with 6 DOF:
- Joint 1 (Hip Yaw): α0, a0, d1, θ1
- Joint 2 (Hip Roll): α1, a1, d2, θ2
- Joint 3 (Hip Pitch): α2, a2, d3, θ3
- Joint 4 (Knee): α3, a3, d4, θ4
- Joint 5 (Ankle Pitch): α4, a4, d5, θ5
- Joint 6 (Ankle Roll): α5, a5, d6, θ6

### Kinematic Solutions for Humanoid Locomotion

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidKinematics:
    def __init__(self, leg_params):
        """
        Initialize kinematic parameters for a humanoid robot
        
        Args:
            leg_params: Dictionary containing link lengths and joint limits
        """
        self.leg_params = leg_params
        self.hip_offset = leg_params.get('hip_offset', [0.0, 0.1, 0.0])  # Lateral hip offset
        
    def homogeneous_transform(self, theta, d, a, alpha):
        """Calculate homogeneous transformation matrix for DH parameters"""
        st, ct = np.sin(theta), np.cos(theta)
        sa, ca = np.sin(alpha), np.cos(alpha)
        
        T = np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
        return T
    
    def leg_forward_kinematics(self, joint_angles):
        """
        Calculate forward kinematics for a single leg
        
        Args:
            joint_angles: List of 6 joint angles [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
        
        Returns:
            Position [x, y, z] and orientation (as quaternion) of the foot
        """
        # Simplified model - in practice, each joint would have its own DH parameters
        # This is a conceptual implementation
        
        # Base transformation (hip position)
        T_base = np.eye(4)
        T_base[0:3, 3] = [0, self.leg_params['hip_y_offset'], self.leg_params['hip_z_offset']]
        
        # Calculate transformations for each joint
        T_total = T_base
        for i, angle in enumerate(joint_angles):
            # This is a simplified representation
            # In practice, each joint would modify T_total differently
            joint_transform = self.get_joint_transform(i, angle)
            T_total = T_total @ joint_transform
        
        # Extract position and orientation
        position = T_total[0:3, 3]
        rotation_matrix = T_total[0:3, 0:3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()
        
        return position, quaternion
    
    def get_joint_transform(self, joint_idx, angle):
        """Get transformation matrix for a specific joint"""
        # This is a simplified model - a complete implementation would use actual DH parameters
        T = np.eye(4)
        
        if joint_idx == 0:  # Hip Yaw
            T[0:3, 0:3] = R.from_euler('z', angle).as_matrix()
        elif joint_idx == 1:  # Hip Roll
            T[0:3, 0:3] = R.from_euler('x', angle).as_matrix()
        elif joint_idx == 2:  # Hip Pitch
            T[0:3, 0:3] = R.from_euler('y', angle).as_matrix()
        elif joint_idx == 3:  # Knee
            T[0:3, 0:3] = R.from_euler('y', angle).as_matrix()
        elif joint_idx == 4:  # Ankle Pitch
            T[0:3, 0:3] = R.from_euler('y', angle).as_matrix()
        elif joint_idx == 5:  # Ankle Roll
            T[0:3, 0:3] = R.from_euler('x', angle).as_matrix()
        
        return T
    
    def calculate_inverse_kinematics(self, target_position, target_orientation, chain_type='leg'):
        """
        Calculate inverse kinematics for a kinematic chain
        
        Args:
            target_position: Desired [x, y, z] position of end effector
            target_orientation: Desired orientation (quaternion)
            chain_type: Type of chain ('leg', 'arm', etc.)
        
        Returns:
            Joint angles that achieve the target pose (or closest approximation)
        """
        if chain_type == 'leg':
            # Simplified inverse kinematics for leg
            # In reality, this would be much more complex and possibly iterative
            return self.calculate_leg_ik(target_position, target_orientation)
        elif chain_type == 'arm':
            # Simplified inverse kinematics for arm
            return self.calculate_arm_ik(target_position, target_orientation)
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")
    
    def calculate_leg_ik(self, target_pos, target_quat):
        """Simplified inverse kinematics for leg (conceptual implementation)"""
        # This is a simplified implementation
        # In practice, humanoid leg IK involves complex geometric and numerical solutions
        
        # Calculate hip-to-foot vector
        hip_to_foot = target_pos - np.array([0, self.leg_params['hip_y_offset'], self.leg_params['hip_z_offset']])
        
        # Simplified solution: ignore hip rotation for now
        # In reality, hip yaw, roll, and pitch would be calculated based on foot position
        hip_yaw = np.arctan2(hip_to_foot[1], hip_to_foot[0])
        
        # Calculate knee angle based on leg length
        xy_dist = np.sqrt(hip_to_foot[0]**2 + hip_to_foot[1]**2)
        z_dist = hip_to_foot[2]
        leg_length = np.sqrt(xy_dist**2 + z_dist**2)
        
        # Assuming 2-link leg model (thigh and shank)
        thigh_length = self.leg_params['thigh_length']
        shank_length = self.leg_params['shank_length']
        
        # Law of cosines to calculate knee angle
        cos_knee = (thigh_length**2 + shank_length**2 - leg_length**2) / (2 * thigh_length * shank_length)
        knee_angle = np.pi - np.arccos(np.clip(cos_knee, -1, 1))
        
        # Calculate hip pitch to reach target
        # This is simplified and in reality involves more complex calculations
        hip_pitch = np.arctan2(z_dist, xy_dist)  # This is a very simplified approximation
        
        # Ankle angles to achieve desired foot orientation
        # Would need to consider current leg configuration
        ankle_pitch = 0.0
        ankle_roll = 0.0
        
        joint_angles = [
            hip_yaw,      # Hip Yaw
            0.0,          # Hip Roll (simplified)
            hip_pitch,    # Hip Pitch
            knee_angle,   # Knee
            ankle_pitch,  # Ankle Pitch
            ankle_roll    # Ankle Roll
        ]
        
        return np.array(joint_angles)

# Example usage
if __name__ == "__main__":
    # Define leg parameters
    leg_params = {
        'hip_y_offset': 0.1,      # Lateral hip offset
        'hip_z_offset': 0.9,      # Hip height above ankle
        'thigh_length': 0.45,     # Length from hip to knee
        'shank_length': 0.45,     # Length from knee to ankle
        'foot_length': 0.25       # Length of foot
    }
    
    # Initialize kinematics
    kin = HumanoidKinematics(leg_params)
    
    # Calculate forward kinematics
    joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial position
    pos, quat = kin.leg_forward_kinematics(joint_angles)
    print(f"Foot position: {pos}, orientation: {quat}")
    
    # Calculate inverse kinematics
    target_pos = [0.0, 0.0, -0.8]  # Target foot position (x=0, y=0, z=-0.8m)
    target_quat = [0, 0, 0, 1]     # Identity orientation
    ik_angles = kin.calculate_inverse_kinematics(target_pos, target_quat, 'leg')
    print(f"IK joint angles: {ik_angles}")
```

### Balance Control Implementation

```python
class BalanceController:
    def __init__(self, robot_params):
        """
        Initialize balance controller for humanoid robot
        
        Args:
            robot_params: Dictionary containing robot physical parameters
        """
        self.mass = robot_params.get('total_mass', 50.0)  # kg
        self.gravity = 9.81  # m/s^2
        self.support_polygon = []  # vertices of support polygon (foot area)
        self.com_threshold = robot_params.get('com_threshold', 0.05)  # 5cm threshold
        
    def calculate_support_polygon(self, left_foot_pos, right_foot_pos):
        """Calculate support polygon based on foot positions"""
        # Simplified support polygon as a rectangle between feet
        # In practice, this would be based on actual foot geometry
        
        # Calculate average foot position for double support
        avg_x = (left_foot_pos[0] + right_foot_pos[0]) / 2.0
        avg_y = (left_foot_pos[1] + right_foot_pos[1]) / 2.0
        
        # Calculate polygon vertices (simplified as a box)
        width = abs(left_foot_pos[1] - right_foot_pos[1])  # distance between feet
        length = 0.3  # approximate foot length
        
        vertices = [
            [avg_x - length/2, avg_y - width/2, 0],
            [avg_x + length/2, avg_y - width/2, 0],
            [avg_x + length/2, avg_y + width/2, 0],
            [avg_x - length/2, avg_y + width/2, 0]
        ]
        
        return vertices
    
    def is_stable(self, com_position):
        """Check if the robot is in a stable state"""
        # Calculate projection of CoM onto ground plane
        com_proj = [com_position[0], com_position[1], 0]
        
        # Check if CoM projection is within support polygon
        # Using a simplified point-in-polygon test
        return self.point_in_polygon(com_proj, self.support_polygon)
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon (simplified implementation)"""
        # This is a conceptual implementation
        # Real application would use proper geometric algorithms
        
        x, y = point[0], point[1]
        
        # Simplified: check if point is within bounding box of polygon
        xs = [v[0] for v in polygon]
        ys = [v[1] for v in polygon]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def compute_zmp(self, com_position, com_velocity, com_acceleration):
        """Calculate Zero Moment Point from CoM information"""
        # ZMP = [CoM_x - (g/h) * CoM_accel_x, CoM_y - (g/h) * CoM_accel_y]
        # where h is the height of the CoM above ground
        
        h = com_position[2]  # height of CoM above ground
        if h <= 0:
            h = 0.001  # prevent division by zero
        
        zmp_x = com_position[0] - (self.gravity / h) * com_acceleration[0]
        zmp_y = com_position[1] - (self.gravity / h) * com_acceleration[1]
        
        return [zmp_x, zmp_y, 0]
    
    def adjust_posture_for_balance(self, current_com, desired_com):
        """Adjust joint angles to maintain balance"""
        # This would involve complex control algorithms
        # such as PID controllers, model predictive control, etc.
        
        # Simplified approach: adjust based on CoM error
        com_error = np.array(desired_com) - np.array(current_com)
        
        # Return a compensation value for joints
        # In practice, this would be translated to joint angle adjustments
        compensation = 0.5 * com_error  # proportional control (simplified)
        
        return compensation
```

### Walking Pattern Generation

```python
class WalkingPatternGenerator:
    def __init__(self, step_params):
        """
        Initialize walking pattern generator
        
        Args:
            step_params: Dictionary containing walking parameters
        """
        self.step_length = step_params.get('step_length', 0.3)  # m
        self.step_width = step_params.get('step_width', 0.2)   # m
        self.step_height = step_params.get('step_height', 0.05) # m
        self.step_duration = step_params.get('step_duration', 1.0)  # s
        self.zmp_margin = step_params.get('zmp_margin', 0.02)  # safety margin
        
    def generate_walking_pattern(self, num_steps, start_pos=[0, 0, 0]):
        """Generate a walking pattern for the specified number of steps"""
        # This is a conceptual implementation of a walking pattern generator
        # A complete implementation would use the inverted pendulum model or similar
        
        pattern = []
        
        # Initial position
        current_pos = list(start_pos)
        
        for i in range(num_steps):
            # Determine if this is a right or left step
            is_right_step = (i % 2 == 1)
            
            # Calculate foot position for this step
            if is_right_step:
                foot_pos = [
                    current_pos[0] + self.step_length,
                    current_pos[1] - self.step_width/2,  # Right foot to the right
                    current_pos[2]
                ]
            else:
                foot_pos = [
                    current_pos[0] + self.step_length,
                    current_pos[1] + self.step_width/2,  # Left foot to the left
                    current_pos[2]
                ]
            
            # Generate step trajectory (simplified as a 3-point arc)
            step_trajectory = self.generate_step_trajectory(
                current_pos, foot_pos, self.step_height, self.step_duration
            )
            
            # Update current position for next step
            current_pos[0] = foot_pos[0]  # Move forward
            current_pos[1] = 0  # Return to center line
            
            pattern.append({
                'step_number': i + 1,
                'foot_position': foot_pos,
                'is_right_foot': is_right_step,
                'trajectory': step_trajectory
            })
        
        return pattern
    
    def generate_step_trajectory(self, start_pos, end_pos, step_height, duration):
        """Generate a trajectory for a single step"""
        # Create a 3D trajectory for the foot movement
        # This is simplified to a parabolic arc in the z-direction
        # and linear movement in x and y directions
        
        trajectory = []
        num_points = 20  # number of points in the trajectory
        
        for i in range(num_points):
            t = i / (num_points - 1)  # normalized time (0 to 1)
            
            # Linear interpolation for x and y
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            
            # Parabolic interpolation for z (to create step motion)
            # Peak at t = 0.5
            z = start_pos[2] + 4 * step_height * t * (1 - t)
            
            trajectory.append([x, y, z])
        
        return trajectory
```

## Advanced Topics

### Whole-Body Kinematics

Humanoid robots often use whole-body kinematic controllers that coordinate all joints simultaneously to achieve multiple tasks:

- Maintaining balance
- Following a walking pattern
- Controlling arm movements
- Keeping head oriented toward targets

This is typically formulated as an optimization problem that minimizes a cost function while satisfying constraints.

### Kinematic Redundancy

Humanoid robots have redundant DOF that can be used for secondary objectives like:
- Avoiding joint limits
- Maintaining comfortable postures
- Avoiding obstacles
- Optimizing energy consumption

## Troubleshooting Common Issues

### Singularity Problems

Humanoid robots can experience kinematic singularities where the Jacobian matrix becomes rank-deficient. To address this:
- Regularly check for singular configurations
- Use damped least squares methods
- Implement joint limit avoidance
- Plan paths that avoid singular regions

### Balance Instability

If your humanoid robot experiences balance issues:
- Verify CoM calculation accuracy
- Check support polygon computation
- Adjust control gains appropriately
- Verify sensor calibration (IMU, force/torque sensors)

### Inverse Kinematics Convergence

If IK solutions fail to converge:
- Check for joint limit violations
- Verify that target poses are reachable
- Use better initial guesses for iterative solvers
- Consider using multiple different IK algorithms

## Best Practices

### Model Accuracy

- Use accurate physical models of your robot
- Regularly update kinematic parameters based on measurements
- Account for actual joint limits and velocity limits
- Include compliance effects in high-precision applications

### Computational Efficiency

- Precompute Jacobians when possible
- Use efficient numerical methods
- Consider hybrid analytical/numerical approaches
- Implement caching for repeated calculations

### Safety

- Always check for collisions during motion planning
- Verify balance before executing motions
- Implement emergency stop procedures
- Use conservative control parameters initially

## Hands-on Exercise

1. **Kinematic Analysis**: Analyze the kinematic chain of a humanoid robot design. Determine the total degrees of freedom and identify which joints contribute to balance vs. manipulation.

2. **Forward Kinematics Implementation**: Implement forward kinematics for a simplified 6-DOF leg model using the DH parameters approach. Verify your implementation by comparing end-effector positions for known joint angles.

3. **Balance Simulation**: Create a simulation of a humanoid robot standing on one foot. Calculate the center of mass position and verify it remains within the support polygon of the supporting foot.

4. **Walking Pattern Generation**: Implement a simple walking pattern generator for a humanoid robot. Generate step positions for a 10-step forward walk and visualize the resulting path.

5. **Inverse Kinematics Challenge**: Implement an inverse kinematics solution for positioning a humanoid robot's foot at a specific location. Test with various target positions to verify reachability.

## Key Takeaways

- Humanoid kinematics involve complex multi-chain coordination
- Balance is a primary concern in humanoid robotics
- Forward and inverse kinematics must consider all links simultaneously
- Efficient algorithms are essential for real-time control
- Safety and stability constraints are critical in motion planning
- Modeling accuracy directly impacts performance

## Further Reading

- "Introduction to Humanoid Robotics" by Shumskaya and Kajita
- "Humanoid Robotics" by Herr and Bellman
- "Kinematic Modeling, Reduction and Planning of Humanoid Robots" - Technical papers
- ROS MoveIt! documentation for motion planning

## Next Steps

Continue to Chapter 2: Bipedal Locomotion to explore the dynamics and control strategies for humanoid walking.