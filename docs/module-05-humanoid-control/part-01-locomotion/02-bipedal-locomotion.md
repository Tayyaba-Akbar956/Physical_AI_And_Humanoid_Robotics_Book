---
sidebar_position: 2
title: Bipedal Locomotion
---

# Bipedal Locomotion

This chapter delves into the mechanics and control strategies for bipedal walking in humanoid robots. Bipedal locomotion is one of the most challenging aspects of humanoid robotics, requiring precise balance control, dynamic stability, and coordinated movement of multiple body segments. The chapter covers both theoretical foundations and practical implementation approaches.

## Learning Objectives

- Understand the dynamics of bipedal walking
- Implement control strategies for stable bipedal locomotion
- Analyze the differences between human and robotic walking
- Apply dynamic balance concepts to walking control
- Design walking patterns for humanoid robots

## Introduction: The Challenge of Two-Legged Walking

Bipedal locomotion is one of the most complex motor tasks humans perform effortlessly, but it presents significant challenges for robots. Unlike wheeled vehicles that can maintain stability through continuous contact with the ground, bipedal robots must constantly adjust their posture to maintain balance while moving. This requires sophisticated control algorithms and precise actuator coordination.

Human walking involves several phases:
- **Double Support**: Both feet are in contact with the ground
- **Single Support**: Only one foot is in contact with the ground
- **Swing Phase**: One foot is moving forward while the other supports the body
- **Transfer Phase**: Weight is transferred from one foot to the other

During walking, the human body constantly makes micro-adjustments to maintain balance, using sensory feedback from the vestibular system, visual input, and proprioceptive sensors.

### Key Challenges in Robotic Bipedal Locomotion

1. **Dynamic Balance**: Maintaining stability during the single-support phase
2. **Ground Impact**: Managing the forces generated when the foot contacts the ground
3. **Energy Efficiency**: Minimizing energy consumption while walking
4. **Terrain Adaptation**: Adjusting to uneven surfaces and obstacles
5. **Robustness**: Maintaining walking even with perturbations and disturbances

## Core Concepts

### Walking Phases

**Double Support Phase**:
- Both feet are in contact with the ground
- The center of mass transitions from one foot to the other
- Provides stability during weight transfer
- Generally shorter in faster walking

**Single Support Phase**:
- Only one foot is in contact with the ground
- The body's center of mass follows an inverted pendulum motion
- Requires active balance control
- Forms the majority of the walking cycle

### Dynamic Balance Models

**Inverted Pendulum Model**:
The simplest model for bipedal walking treats the robot as an inverted pendulum pivoting over the stance foot. This model captures the essential dynamics of balance control:
- Center of mass moves in an arc above the support foot
- Stability depends on controlling the CoM position relative to the support point
- Can be linearized around the upright position

**Linear Inverted Pendulum (LIP)**:
A more practical model that assumes the center of mass height remains constant:
- CoM moves at a constant height above the ground
- Simplifies the mathematics while retaining essential dynamics
- Allows for analytical solutions to balance control

**Capture Point**:
A key concept in bipedal balance - the point on the ground where the robot can step to stop its current momentum:
- If the CoM velocity vector passes through the capture point, the robot can stop
- Determines where the robot should place its foot for stability
- Critical for recovery from disturbances

### Gait Parameters

**Step Length**: Distance between the heel of the stance foot and the heel of the swing foot at initial contact.

**Step Width**: Lateral distance between the feet during walking.

**Step Height**: Maximum height of the swing foot above the ground during the swing phase.

**Stride Length**: Distance traveled by the body between two consecutive heel contacts of the same foot.

**Walking Speed**: Average forward velocity of the robot.

**Step Frequency**: Number of steps per unit time.

## Mathematical Framework

### Inverted Pendulum Dynamics

The dynamics of the inverted pendulum model can be described by:

L * d²θ/dt² = g * sin(θ)

Where:
- L is the pendulum length (CoM height)
- θ is the angle from vertical
- g is the acceleration due to gravity

For small angles, this linearizes to:
L * d²θ/dt² = g * θ

### Zero-Moment Point (ZMP)

The ZMP is a critical concept in bipedal locomotion:

ZMP_x = CoM_x - (h/g) * CoM_acc_x
ZMP_y = CoM_y - (h/g) * CoM_acc_y

Where:
- CoM_x, CoM_y are the center of mass coordinates
- h is the CoM height above the ground
- CoM_acc_x, CoM_acc_y are the CoM acceleration components
- g is the acceleration due to gravity

For stable walking, the ZMP must remain within the support polygon defined by the feet.

### Linear Inverted Pendulum Model (LIPM)

In the LIPM, the center of mass height remains constant:

CoM(t) = ZMP + (CoM(0) - ZMP) * cosh(t/τ) + CoM_dot(0) * τ * sinh(t/τ)

Where:
- τ = √(h/g) is the time constant of the pendulum
- CoM(0) and CoM_dot(0) are the initial position and velocity

## Practical Implementation

### ZMP-Based Walking Pattern Generation

```python
import numpy as np
from scipy import integrate

class ZMPBasedWalker:
    def __init__(self, robot_params):
        """
        Initialize ZMP-based walking controller
        
        Args:
            robot_params: Dictionary containing robot physical parameters
        """
        self.robot_height = robot_params.get('robot_height', 0.8)  # height of CoM
        self.step_length = robot_params.get('step_length', 0.3)   # desired step length
        self.step_width = robot_params.get('step_width', 0.2)    # lateral foot distance
        self.walk_period = robot_params.get('walk_period', 1.0)  # time for one step
        self.gravity = 9.81
        self.tau = np.sqrt(self.robot_height / self.gravity)  # time constant
        
        # Initialize walking state
        self.current_support_foot = 'right'  # Start with right foot support
        self.step_count = 0
        self.current_pos = [0.0, 0.0, 0.0]  # x, y, z position
        
    def generate_zmp_trajectory(self, step_time, start_pos, target_pos):
        """
        Generate ZMP trajectory for a single step
        
        Args:
            step_time: Time duration of the step
            start_pos: Starting CoM position
            target_pos: Target CoM position after step
        
        Returns:
            Array of ZMP positions over time
        """
        # Simplified ZMP trajectory generation
        # In practice, this would use more sophisticated methods like spline interpolation
        # or numerical optimization
        
        # Create time vector
        dt = 0.01  # 10ms time steps
        time_vec = np.arange(0, step_time, dt)
        
        # For a simple walk, ZMP moves from one foot to the other
        # This is a simplified model where ZMP moves linearly between foot positions
        if self.current_support_foot == 'right':
            # Right foot support, ZMP starts at right foot position
            zmp_start = [start_pos[0], start_pos[1] - self.step_width/2]
            zmp_end = [target_pos[0], target_pos[1] + self.step_width/2]  # Next left foot
        else:
            # Left foot support, ZMP starts at left foot position
            zmp_start = [start_pos[0], start_pos[1] + self.step_width/2]
            zmp_end = [target_pos[0], target_pos[1] - self.step_width/2]  # Next right foot
        
        # Linear interpolation for ZMP trajectory
        zmp_traj = []
        for t in time_vec:
            ratio = t / step_time
            zmp_x = zmp_start[0] + ratio * (zmp_end[0] - zmp_start[0])
            zmp_y = zmp_start[1] + ratio * (zmp_end[1] - zmp_start[1])
            zmp_traj.append([zmp_x, zmp_y])
        
        return np.array(zmp_traj), time_vec
    
    def calculate_com_trajectory(self, zmp_trajectory, time_vector):
        """
        Calculate CoM trajectory from ZMP trajectory using LIPM
        
        Args:
            zmp_trajectory: Array of ZMP positions over time
            time_vector: Time values corresponding to ZMP positions
        
        Returns:
            CoM trajectory over time
        """
        # Solve the LIPM equation: CoM(t) = ZMP(t) + (CoM(0) - ZMP(0)) * cosh(t/τ) + CoM_dot(0) * τ * sinh(t/τ)
        # This is a simplified implementation
        
        com_trajectory = []
        
        # Initial conditions (would be determined from robot state)
        initial_com = np.array([0.0, 0.0])  # Initial CoM position (x, y)
        initial_com_vel = np.array([0.0, 0.0])  # Initial CoM velocity
        
        # For the first point
        com_trajectory.append(initial_com.copy())
        
        # Calculate CoM at each time step based on ZMP
        for i in range(1, len(time_vector)):
            dt = time_vector[i] - time_vector[i-1]
            
            # In a full implementation, this would solve the LIPM differential equation
            # Here's a simplified approach using discrete integration
            
            # This is still a simplified version - a complete implementation would
            # solve the differential equation of the LIPM model
            current_zmp = zmp_trajectory[i]
            prev_com = com_trajectory[i-1]
            
            # Approximate using the LIPM relationship
            # CoM acceleration = g/h * (CoM - ZMP)
            if i == 1:  # Use initial conditions
                prev_com = initial_com
                com_vel = initial_com_vel
            else:
                com_vel = (com_trajectory[i-1] - com_trajectory[i-2]) / dt if i > 1 else initial_com_vel
            
            # Calculate acceleration based on CoM-ZMP difference
            com_acc = (self.gravity / self.robot_height) * (prev_com - current_zmp)
            
            # Update CoM position using kinematic equations
            new_com = prev_com + com_vel * dt + 0.5 * com_acc * dt**2
            com_trajectory.append(new_com)
            
        return np.array(com_trajectory)
    
    def generate_foot_steps(self, walk_distance, step_count):
        """
        Generate a sequence of foot positions for walking
        
        Args:
            walk_distance: Total distance to walk
            step_count: Number of steps to take
        
        Returns:
            List of foot positions (x, y, z) for each step
        """
        foot_steps = []
        
        # Start position
        current_x = 0
        current_y = 0
        
        for i in range(step_count):
            # Determine which foot to place
            if i % 2 == 0:  # Even steps: place left foot (if starting with right support)
                foot_x = current_x + self.step_length
                foot_y = current_y + self.step_width / 2
                foot_type = 'left'
            else:  # Odd steps: place right foot
                foot_x = current_x + self.step_length
                foot_y = current_y - self.step_width / 2
                foot_type = 'right'
            
            # Update current position for next step
            current_x = foot_x
            current_y = 0  # Reset to center line
            
            foot_steps.append({
                'step_number': i + 1,
                'type': foot_type,
                'position': [foot_x, foot_y, 0],
                'time': (i + 1) * self.walk_period
            })
        
        return foot_steps
    
    def walk_cycle(self, step_number, next_support_pos):
        """
        Execute a single walking step
        
        Args:
            step_number: The current step number
            next_support_pos: Position of the next support foot
        
        Returns:
            Trajectory for the step
        """
        # Generate ZMP trajectory for this step
        zmp_trajectory, time_vector = self.generate_zmp_trajectory(
            self.walk_period, 
            self.current_pos[:2], 
            next_support_pos[:2]
        )
        
        # Calculate CoM trajectory from ZMP
        com_trajectory = self.calculate_com_trajectory(zmp_trajectory, time_vector)
        
        # Switch support foot
        self.current_support_foot = 'left' if self.current_support_foot == 'right' else 'right'
        
        # Update current position based on last CoM position
        self.current_pos[0] = com_trajectory[-1][0]
        self.current_pos[1] = com_trajectory[-1][1]
        
        return {
            'step_number': step_number,
            'zmp_trajectory': zmp_trajectory,
            'com_trajectory': com_trajectory,
            'time_vector': time_vector,
            'current_pos': self.current_pos.copy()
        }

# Example usage
if __name__ == "__main__":
    # Robot parameters
    robot_params = {
        'robot_height': 0.8,      # Height of CoM in meters
        'step_length': 0.3,       # Step length in meters
        'step_width': 0.2,        # Step width in meters
        'walk_period': 1.0        # Time per step in seconds
    }
    
    # Initialize walker
    walker = ZMPBasedWalker(robot_params)
    
    # Generate walking pattern for 5 steps
    steps = walker.generate_foot_steps(walk_distance=1.5, step_count=5)
    
    print("Foot steps planned:")
    for step in steps:
        print(f"Step {step['step_number']} ({step['type']}): {step['position']} at t={step['time']:.2f}s")
    
    # Execute first step
    next_support_pos = steps[0]['position']  # Position for next support foot
    step_result = walker.walk_cycle(1, next_support_pos)
    
    print(f"\nStep 1 trajectory calculated with {len(step_result['time_vector'])} time steps")
    print(f"Final CoM position: {step_result['com_trajectory'][-1]}")
```

### Foot Trajectory Generation

```python
class FootTrajectoryGenerator:
    def __init__(self, robot_params):
        """
        Generate trajectories for foot movement during walking
        
        Args:
            robot_params: Dictionary containing robot parameters
        """
        self.step_height = robot_params.get('step_height', 0.05)  # foot clearance in meters
        self.swing_duration_ratio = robot_params.get('swing_ratio', 0.35)  # portion of step for swing
        self.foot_length = robot_params.get('foot_length', 0.25)    # length of foot in meters
        
    def generate_foot_trajectory(self, lift_pos, land_pos, step_duration):
        """
        Generate a trajectory for the foot from lift to landing position
        
        Args:
            lift_pos: Position where foot lifts off (x, y, z)
            land_pos: Position where foot lands (x, y, z)
            step_duration: Total duration of the step
        
        Returns:
            Array of foot positions over time
        """
        dt = 0.01  # 10ms time steps
        time_steps = int(step_duration / dt)
        time_vector = np.linspace(0, step_duration, time_steps)
        
        trajectory = []
        
        # Calculate swing phase duration and double support phase duration
        swing_duration = step_duration * self.swing_duration_ratio
        double_support_duration = step_duration * (1 - self.swing_duration_ratio) / 2
        
        for i, t in enumerate(time_vector):
            # Determine which phase of step we're in
            if t <= double_support_duration:
                # Initial double support phase - foot remains at old position
                pos = lift_pos.copy()
            elif t >= step_duration - double_support_duration:
                # Final double support phase - foot remains at new position
                pos = land_pos.copy()
            else:
                # Swing phase
                # Calculate normalized time within swing phase
                swing_t = (t - double_support_duration) / swing_duration
                
                # Calculate x and y positions (linear interpolation)
                x = lift_pos[0] + swing_t * (land_pos[0] - lift_pos[0])
                y = lift_pos[1] + swing_t * (land_pos[1] - lift_pos[1])
                
                # Calculate z position (parabolic trajectory for foot clearance)
                # Peak at middle of swing phase
                z = lift_pos[2] + 4 * self.step_height * swing_t * (1 - swing_t)
                
                pos = [x, y, z]
            
            trajectory.append(pos)
        
        return np.array(trajectory), time_vector

# Example usage of foot trajectory
if __name__ == "__main__":
    # Initialize foot trajectory generator
    params = {'step_height': 0.05, 'swing_ratio': 0.35, 'foot_length': 0.25}
    foot_gen = FootTrajectoryGenerator(params)
    
    # Generate trajectory for a step
    lift_pos = [0.0, -0.1, 0.0]  # Starting foot position
    land_pos = [0.3, 0.1, 0.0]   # Landing foot position
    step_dur = 1.0               # Step duration in seconds
    
    foot_traj, time_vec = foot_gen.generate_foot_trajectory(lift_pos, land_pos, step_dur)
    
    print(f"Generated foot trajectory with {len(foot_traj)} points")
    print(f"Foot lifts at: {foot_traj[0]}")
    print(f"Foot lands at: {foot_traj[-1]}")
    print(f"Max foot height: {np.max(foot_traj[:, 2]):.3f}m")
```

### Balance Control Implementation

```python
class BalanceController:
    def __init__(self, robot_params):
        """
        Advanced balance controller for humanoid robots
        
        Args:
            robot_params: Dictionary containing robot parameters
        """
        self.mass = robot_params.get('total_mass', 50.0)
        self.com_height = robot_params.get('robot_height', 0.8)
        self.gravity = 9.81
        self.tau = np.sqrt(self.com_height / self.gravity)
        
        # PID controller parameters for balance
        self.kp_com = robot_params.get('kp_com', 40.0)    # Proportional gain for CoM
        self.kd_com = robot_params.get('kd_com', 20.0)    # Derivative gain for CoM
        self.kp_foot = robot_params.get('kp_foot', 10.0)  # Proportional gain for foot placement
        self.control_dt = 0.01  # 10ms control loop
        
        # State estimation
        self.current_com = np.array([0.0, 0.0, self.com_height])
        self.current_com_vel = np.array([0.0, 0.0, 0.0])
        self.current_zmp = np.array([0.0, 0.0, 0.0])
    
    def update_state(self, measured_com, measured_com_vel):
        """Update controller with current state measurements"""
        self.current_com = measured_com
        self.current_com_vel = measured_com_vel
        
        # Estimate ZMP from current CoM state
        # ZMP = CoM - (h/g) * CoM_acc
        # Since we don't have acceleration directly, we estimate it or use it from state
        # This is a simplified approach - in practice, ZMP would be measured with force sensors
        self.current_zmp = self.estimate_zmp()
    
    def estimate_zmp(self):
        """Estimate Zero Moment Point from current CoM state"""
        # This is a simplified estimation
        # In practice, ZMP is typically measured using force/torque sensors in the feet
        return np.array([self.current_com[0], self.current_com[1], 0.0])
    
    def compute_balance_control(self, desired_zmp):
        """
        Compute balance control corrections
        
        Args:
            desired_zmp: Target ZMP position for stability
        
        Returns:
            Joint adjustments needed for balance
        """
        # Calculate ZMP error
        zmp_error = desired_zmp[:2] - self.current_zmp[:2]
        
        # Calculate CoM error
        com_error = desired_zmp[:2] - self.current_com[:2]
        
        # PID control for CoM position
        com_control = self.kp_com * com_error  # + self.kd_com * com_velocity_error
        
        # Calculate corrective forces needed
        corrective_force = self.mass * self.gravity * com_control / self.com_height
        
        # Convert to joint angle adjustments (conceptual mapping)
        # In reality, this mapping would be more complex and specific to the robot
        joint_corrections = {
            'hip_roll': corrective_force[1] * 0.1,  # Lateral force -> hip roll
            'hip_pitch': corrective_force[0] * 0.1, # Forward force -> hip pitch
            'ankle_roll': corrective_force[1] * 0.2, # Lateral force -> ankle roll
            'ankle_pitch': corrective_force[0] * 0.2 # Forward force -> ankle pitch
        }
        
        return joint_corrections
    
    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate the capture point for balance recovery
        
        Args:
            com_pos: Current center of mass position
            com_vel: Current center of mass velocity
        
        Returns:
            Capture point position where robot should step to stop
        """
        # Capture point = CoM position + CoM velocity * sqrt(h/g)
        capture_point = com_pos[:2] + com_vel[:2] * self.tau
        return capture_point
    
    def adjust_step_location(self, current_support_pos, com_pos, com_vel):
        """
        Adjust planned step location based on balance requirements
        
        Args:
            current_support_pos: Current support foot position
            com_pos: Current CoM position
            com_vel: Current CoM velocity
        
        Returns:
            Adjusted step location for next foot
        """
        # Calculate where we need to step to maintain balance
        capture_point = self.calculate_capture_point(com_pos, com_vel)
        
        # The foot should be placed near the capture point for stability
        # But we also need to consider the natural walking pattern
        desired_step = capture_point
        
        # Check if this step location is reasonable (within leg reach)
        step_distance = np.linalg.norm(desired_step - current_support_pos[:2])
        max_reach = 0.4  # Maximum comfortable step distance
        
        if step_distance > max_reach:
            # Scale the step to maximum comfortable reach
            direction = (desired_step - current_support_pos[:2]) / step_distance
            desired_step = current_support_pos[:2] + direction * max_reach
        
        return desired_step
```

## Walking Control Strategies

### Model Predictive Control (MPC)

MPC for walking involves solving an optimization problem at each time step to determine the best future actions:

```python
class WalkingMPC:
    def __init__(self, prediction_horizon=10):
        self.horizon = prediction_horizon
        self.weights = {
            'zmp_tracking': 10.0,    # Importance of tracking ZMP references
            'com_velocity': 1.0,     # Penalty for CoM velocity
            'step_size': 0.5,        # Penalty for large step sizes
            'control_effort': 0.1    # Penalty for control effort
        }
    
    def solve_optimization(self, current_state, reference_trajectory):
        """
        Solve the MPC optimization problem
        (In practice, this would use a numerical optimization solver)
        """
        # This is a conceptual implementation
        # A real implementation would use quadratic programming or similar
        pass
```

### Preview Control

Preview control uses knowledge of future terrain or walking pattern to anticipate balance adjustments:

```python
class PreviewController:
    def __init__(self, preview_steps=10):
        self.preview_steps = preview_steps
        
    def generate_preview_control(self, future_zmp_ref):
        """
        Generate control inputs based on preview of future ZMP references
        """
        # Uses the preview information to calculate appropriate CoM trajectory
        # that will achieve future ZMP goals
        pass
```

## Advanced Walking Techniques

### Walking on Uneven Terrain

Adapting walking patterns for uneven surfaces requires:
- Terrain height estimation
- Adjusting foot placement
- Modifying leg trajectories
- Adapting step timing

### Turning and Direction Changes

Implementing turning involves:
- Coordinated step placement
- Hip and torso rotation
- Balance during direction changes

## Troubleshooting Common Issues

### Instability During Walking

If your robot experiences walking instability:
- Check ZMP tracking accuracy
- Verify center of mass estimation
- Adjust control gains appropriately
- Ensure sensors are properly calibrated
- Consider adding more conservative control parameters

### Foot Slipping

To address foot slipping:
- Reduce step velocity
- Increase foot-ground friction modeling
- Add force control to maintain contact
- Use larger support polygons

### Excessive Energy Consumption

For high energy usage:
- Optimize step parameters for efficiency
- Implement more natural walking gaits
- Check for unnecessary joint movements
- Consider using passive dynamics where possible

### Asymmetric Walking

If the robot walks asymmetrically:
- Verify sensor calibration
- Check for mechanical asymmetries
- Verify identical control parameters for both legs
- Check for consistent actuator performance

## Best Practices

### Design Considerations

- Design for robustness to model uncertainties
- Include safety margins in balance calculations
- Plan for sensor/actuator failures
- Consider the trade-off between speed and stability

### Control Implementation

- Use sensor fusion for accurate state estimation
- Implement hierarchical control (balance, locomotion, manipulation)
- Include proper state machines for smooth transitions
- Design controllers based on robot's physical capabilities

### Testing and Validation

- Test in simulation before real robot deployment
- Start with small steps and conservative parameters
- Gradually increase walking speed and complexity
- Test on various terrains and conditions

## Hands-on Exercise

1. **ZMP Analysis**: Calculate and plot the ZMP trajectory for a simple walking pattern. Verify that it remains within the support polygon throughout the walk.

2. **Inverted Pendulum Simulation**: Implement a simulation of the inverted pendulum model for bipedal walking. Observe how the CoM moves and the effect of different control parameters.

3. **Capture Point Calculation**: Implement and test the capture point calculation for different CoM positions and velocities. Understand how this value determines recovery steps.

4. **Step Parameter Tuning**: Experiment with different step lengths, widths, and heights to determine their effect on walking stability and efficiency.

5. **Balance Recovery**: Create a simulation where the robot is pushed during walking and must recover balance by adjusting its next step location.

## Key Takeaways

- Bipedal walking requires precise balance control and coordinated movements
- ZMP and Capture Point are fundamental concepts for walking stability
- Walking patterns must be dynamically stable throughout the gait cycle
- Control strategies must account for robot dynamics and environmental conditions
- Simulation and careful testing are crucial for successful implementation
- Energy efficiency is an important consideration in practical systems

## Further Reading

- "Bipedal Locomotion: A Tutorial" - Kajita and Morisawa
- "Humanoid Robotics" - Herr and Bellman
- "Dynamics and Control of Bipedal Walking" - Technical papers
- "Introduction to Humanoid Robotics" - Shumskaya and Kajita

## Next Steps

Continue to Chapter 3: Balance Control to explore advanced techniques for maintaining stability during dynamic movements.