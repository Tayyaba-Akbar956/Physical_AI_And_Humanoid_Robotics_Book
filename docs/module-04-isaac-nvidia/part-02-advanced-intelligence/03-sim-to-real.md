---
sidebar_position: 3
title: Sim-to-Real Transfer
---

# Sim-to-Real Transfer

This chapter explores the crucial challenge of transferring robotic behaviors learned in simulation to real-world robots. While simulation provides safe, efficient, and cost-effective training environments, the ultimate goal is to deploy these learned behaviors on actual hardware. This chapter will cover the techniques and methodologies that bridge the gap between simulation and reality.

## Learning Objectives

- Understand the "reality gap" and its impact on robotic performance
- Master domain randomization techniques to improve sim-to-real transfer
- Learn about system identification to model real-world discrepancies
- Apply adaptive control methods for sim-to-real deployment
- Implement validation techniques to ensure safe real-world deployment

## Introduction to Reality Gap

The reality gap refers to the differences between simulated environments and the real world that can cause behaviors learned in simulation to fail when deployed on actual robots. These differences include:

- **Physical Properties**: Inaccuracies in simulation models of friction, elasticity, mass distributions
- **Sensor Noise**: Differences in sensor behavior between simulation and reality
- **Actuator Dynamics**: Discrepancies in motor response, latency, and control precision
- **Environmental Factors**: Unmodeled external forces, lighting conditions, or air currents

The sim-to-real transfer problem is fundamental in robotics and requires systematic approaches to ensure that simulation-based learning translates to real-world success.

### The Sim-to-Real Pipeline

```
Simulation Training → Domain Randomization → System Identification → Real-World Deployment
```

Each stage of the pipeline addresses specific aspects of the reality gap to improve transfer performance.

## Domain Randomization

Domain randomization is a key technique that improves sim-to-real transfer by training policies on a wide variety of simulation conditions. Instead of learning from a single, potentially inaccurate, simulation model, the robot learns to adapt to a range of possible environments.

### Implementation of Domain Randomization

```python
import numpy as np
import torch
import torch.nn as nn
from isaacgym import gymapi, gymtorch
from isaacgym import gymutil

class DomainRandomizationEnv:
    def __init__(self):
        # Initialize simulation environment
        self.gym = gymapi.Gym()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self._get_sim_params())
        
        # Define randomization ranges
        self.randomization_ranges = {
            'friction': (0.1, 1.0),           # Range for friction coefficients
            'mass_multiplier': (0.8, 1.2),   # Range for mass scaling
            'restitution': (0.0, 0.5),      # Range for bounciness
            'damping': (0.0, 0.1),          # Range for damping coefficients
            'stiffness': (0.8, 1.2)         # Range for joint stiffness
        }
        
        # Initialize randomized parameters
        self.randomized_params = {}
        
    def _get_sim_params(self):
        # Create simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        return sim_params
        
    def randomize_environment(self):
        """
        Apply randomization to environment parameters
        """
        # Randomize friction
        friction_range = self.randomization_ranges['friction']
        self.randomized_params['friction'] = np.random.uniform(
            friction_range[0], friction_range[1]
        )
        
        # Randomize mass multiplier
        mass_range = self.randomization_ranges['mass_multiplier']
        self.randomized_params['mass_multiplier'] = np.random.uniform(
            mass_range[0], mass_range[1]
        )
        
        # Randomize restitution (bounciness)
        restitution_range = self.randomization_ranges['restitution']
        self.randomized_params['restitution'] = np.random.uniform(
            restitution_range[0], restitution_range[1]
        )
        
        # Randomize damping
        damping_range = self.randomization_ranges['damping']
        self.randomized_params['damping'] = np.random.uniform(
            damping_range[0], damping_range[1]
        )
        
        # Randomize stiffness
        stiffness_range = self.randomization_ranges['stiffness']
        self.randomized_params['stiffness'] = np.random.uniform(
            stiffness_range[0], stiffness_range[1]
        )
        
        print(f"Randomized parameters: {self.randomized_params}")
        
    def apply_randomization_to_robot(self, robot_asset):
        """
        Apply randomization parameters to the robot asset
        """
        # Apply mass randomization
        mass_multiplier = self.randomized_params['mass_multiplier']
        
        # Modify the robot asset with randomized parameters
        # This is a simplified representation - actual implementation would
        # modify the asset properties in the simulation
        print(f"Applying mass multiplier: {mass_multiplier}")
        
    def reset_environment(self):
        """
        Reset the environment with new randomization
        """
        self.randomize_environment()
        # Reset robot position, velocities, etc.
        print("Environment reset with new randomization")

# Example usage for sim-to-real transfer
def train_with_domain_randomization():
    env = DomainRandomizationEnv()
    
    # Train for multiple episodes with different randomizations
    for episode in range(1000):  # In practice, this would run for many more episodes
        if episode % 100 == 0:  # Randomize every 100 episodes
            env.reset_environment()
        
        # Run training step
        print(f"Training episode {episode}")
        
    print("Training completed with domain randomization")

# Run the example
if __name__ == "__main__":
    train_with_domain_randomization()
```

### Advanced Domain Randomization Techniques

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class AdaptiveDomainRandomization:
    def __init__(self, initial_ranges, adaptation_rate=0.01):
        """
        Adaptive domain randomization that modifies randomization parameters
        based on the performance discrepancy between simulation and reality.
        
        Args:
            initial_ranges: Dict of parameter names to (min, max) tuples
            adaptation_rate: Rate at which to adjust randomization ranges
        """
        self.initial_ranges = initial_ranges.copy()
        self.current_ranges = initial_ranges.copy()
        self.adaptation_rate = adaptation_rate
        
        # Track performance metrics
        self.sim_performance = deque(maxlen=100)
        self.real_performance = deque(maxlen=100)
        
    def update_ranges_based_on_performance(self):
        """
        Adjust randomization ranges based on sim vs real performance discrepancy
        """
        if len(self.sim_performance) < 10 or len(self.real_performance) < 10:
            return  # Need sufficient data to make comparison
            
        # Calculate performance discrepancy
        avg_sim_perf = sum(self.sim_performance) / len(self.sim_performance)
        avg_real_perf = sum(self.real_performance) / len(self.real_performance)
        
        discrepancy = abs(avg_sim_perf - avg_real_perf)
        
        # Adjust ranges proportionally to discrepancy
        for param_name, (min_val, max_val) in self.current_ranges.items():
            # Calculate range width
            current_width = max_val - min_val
            # Increase range if discrepancy is high
            new_width = current_width * (1 + discrepancy * self.adaptation_rate)
            
            # Calculate new min/max preserving center
            center = (min_val + max_val) / 2
            new_min = center - new_width / 2
            new_max = center + new_width / 2
            
            # Update range
            self.current_ranges[param_name] = (new_min, new_max)
            
        print(f"Updated ranges: {self.current_ranges}")

    def randomize_parameter(self, param_name):
        """
        Randomize a specific parameter using current ranges
        """
        if param_name not in self.current_ranges:
            raise ValueError(f"Parameter {param_name} not in randomization ranges")
            
        min_val, max_val = self.current_ranges[param_name]
        return np.random.uniform(min_val, max_val)

# Example usage
def example_adaptive_domain_randomization():
    initial_ranges = {
        'friction': (0.1, 1.0),
        'mass_multiplier': (0.8, 1.2),
        'restitution': (0.0, 0.5)
    }
    
    adaptive_dr = AdaptiveDomainRandomization(initial_ranges)
    
    # Simulate training iterations
    for iteration in range(100):
        # Simulate collecting performance data
        sim_perf = np.random.normal(0.8, 0.1)  # Simulated performance
        real_perf = np.random.normal(0.6, 0.1)  # Real performance (lower due to reality gap)
        
        adaptive_dr.sim_performance.append(sim_perf)
        adaptive_dr.real_performance.append(real_perf)
        
        # Occasionally update ranges based on performance
        if iteration % 10 == 0:
            adaptive_dr.update_ranges_based_on_performance()
            
        # Randomize parameters for next iteration
        friction = adaptive_dr.randomize_parameter('friction')
        mass_mult = adaptive_dr.randomize_parameter('mass_multiplier')
        
        print(f"Iteration {iteration}: friction={friction:.3f}, mass_mult={mass_mult:.3f}")

if __name__ == "__main__":
    example_adaptive_domain_randomization()
```

## System Identification for Reality Gap Reduction

System identification involves characterizing the actual physical system to better understand and model the differences between simulation and reality.

### Parameter Estimation

```python
import numpy as np
from scipy.optimize import minimize
from scipy import signal

class SystemIdentifier:
    def __init__(self):
        # True parameters (unknown in reality, but we'll estimate them)
        self.true_params = {
            'mass': 1.0,      # kg
            'friction': 0.1,  # N*s/m
            'gravity': 9.81   # m/s^2
        }
        
        # Estimated parameters (what we'll try to find)
        self.estimated_params = {
            'mass': 0.8,      # Starting estimate
            'friction': 0.05, # Starting estimate
            'gravity': 9.8    # Starting estimate
        }
        
        # Data storage
        self.inputs = []
        self.outputs = []
        
    def simulate_system(self, params, input_signal, dt=0.01):
        """
        Simulate a simple system: mass-spring-damper
        
        Equation: m*ẍ + b*ẋ + k*x = F
        
        For our case: m*ẍ + b*ẋ = F - m*g (simplified)
        """
        time_steps = len(input_signal)
        positions = np.zeros(time_steps)
        velocities = np.zeros(time_steps)
        
        mass = params['mass']
        friction = params['friction']
        gravity = params['gravity']
        
        for i in range(1, time_steps):
            # Calculate acceleration: ẍ = (F - b*ẋ - m*g) / m
            acceleration = (input_signal[i] - friction * velocities[i-1] - mass * gravity) / mass
            # Update velocity: ẋ = ẋ + ẍ*dt
            velocities[i] = velocities[i-1] + acceleration * dt
            # Update position: x = x + ẋ*dt
            positions[i] = positions[i-1] + velocities[i] * dt
            
        return positions, velocities
    
    def collect_data(self, input_signal, dt=0.01):
        """
        Collect data from the true system (with noise)
        """
        # Get response from true system
        true_positions, true_velocities = self.simulate_system(
            self.true_params, input_signal, dt
        )
        
        # Add some measurement noise to simulate real sensors
        noise_level = 0.01
        noisy_positions = true_positions + np.random.normal(0, noise_level, len(true_positions))
        
        # Store data
        self.inputs = np.array(input_signal)
        self.outputs = np.array(noisy_positions)  # Only position measurements for this example
        
        return noisy_positions
    
    def cost_function(self, params_array):
        """
        Cost function for parameter estimation
        """
        # Convert array back to parameter dict
        params = {
            'mass': params_array[0],
            'friction': params_array[1],
            'gravity': params_array[2]
        }
        
        # Simulate with these parameters
        sim_positions, _ = self.simulate_system(params, self.inputs)
        
        # Calculate error with measured data
        error = np.sum((self.outputs - sim_positions)**2)
        
        return error
    
    def identify_system(self):
        """
        Identify system parameters using optimization
        """
        # Initial parameter guess
        initial_guess = [
            self.estimated_params['mass'],
            self.estimated_params['friction'],
            self.estimated_params['gravity']
        ]
        
        # Bounds for parameters
        bounds = [
            (0.1, 5.0),    # mass bounds
            (0.01, 1.0),   # friction bounds
            (9.0, 10.0)    # gravity bounds
        ]
        
        # Optimize
        result = minimize(
            self.cost_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Update estimated parameters
        self.estimated_params['mass'] = result.x[0]
        self.estimated_params['friction'] = result.x[1]
        self.estimated_params['gravity'] = result.x[2]
        
        print(f"True parameters: {self.true_params}")
        print(f"Estimated parameters: {self.estimated_params}")
        print(f"Optimization success: {result.success}")
        print(f"Final cost: {result.fun}")
        
        return result

# Example usage
def example_system_identification():
    identifier = SystemIdentifier()
    
    # Create input signal (e.g., step input or chirp signal)
    t = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz
    input_signal = 10.0 * np.ones_like(t)  # Constant force input
    
    # Collect data from the true system (with noise)
    measured_positions = identifier.collect_data(input_signal)
    
    # Identify system parameters
    result = identifier.identify_system()
    
    return identifier

if __name__ == "__main__":
    example_system_identification()
```

## Adaptive Control for Sim-to-Real Transfer

Adaptive control systems can adjust their behavior to compensate for model uncertainties and environmental changes.

### Model Reference Adaptive Control (MRAC)

```python
import numpy as np
import matplotlib.pyplot as plt

class ModelReferenceAdaptiveController:
    def __init__(self, plant_params, reference_model_params, adaptation_rate=0.1):
        """
        Initialize MRAC controller
        
        Args:
            plant_params: Initial estimates of plant parameters [a, b] in ẋ = ax + bu
            reference_model_params: Parameters for reference model [a_m, b_m]
            adaptation_rate: Rate for parameter adaptation
        """
        self.plant_params = np.array(plant_params)  # [â, b̂]
        self.ref_params = np.array(reference_model_params)  # [a_m, b_m]
        self.adaptation_rate = adaptation_rate
        
        # Controller parameters
        self.k_r = -1.0  # Reference gain
        self.k_y = -1.0  # Output feedback gain
        
        # For adaptation law
        self.theta = np.zeros(2)  # Parameter adjustments [delta_a, delta_b]
        
        # State tracking
        self.state = 0.0
        self.ref_state = 0.0
        self.error = 0.0
        
    def control(self, reference_input, actual_output, dt=0.01):
        """
        Calculate control input using MRAC
        """
        # Calculate reference model state
        self.ref_state += dt * (
            self.ref_params[0] * self.ref_state + 
            self.ref_params[1] * reference_input
        )
        
        # Calculate tracking error
        self.error = self.ref_state - actual_output
        
        # Calculate control input
        control_input = (
            (self.ref_params[0] - self.plant_params[0]) * actual_output +
            (self.ref_params[1] - self.plant_params[1]) * self.k_r * reference_input +
            self.k_r * reference_input - 
            self.k_y * self.error +
            np.dot(self.theta, [actual_output, reference_input])
        )
        
        # Update adaptive parameters
        self._update_parameters(actual_output, reference_input, dt)
        
        return control_input
    
    def _update_parameters(self, y, r, dt):
        """
        Update adaptive parameters using MIT rule
        """
        # Gradient of error with respect to parameters
        phi = np.array([y, self.k_r * r])
        
        # Update theta using MIT rule
        self.theta += self.adaptation_rate * self.error * phi * dt

class AdaptiveRoboticSystem:
    def __init__(self):
        # True plant parameters (unknown to controller)
        self.true_plant_params = np.array([0.5, 1.2])  # [a, b] in ẋ = ax + bu
        self.state = 0.0
        
        # Initialize MRAC with initial estimates
        initial_estimates = [0.3, 1.0]  # Initial estimates
        reference_params = [-2.0, 1.0]  # Stable reference model
        
        self.mrac = ModelReferenceAdaptiveController(
            initial_estimates, reference_params
        )
    
    def step(self, reference_input, dt=0.01):
        """
        Step the adaptive system
        """
        # Get control input from MRAC
        control_input = self.mrac.control(reference_input, self.state, dt)
        
        # Apply control to actual plant (with true parameters)
        state_dot = (
            self.true_plant_params[0] * self.state + 
            self.true_plant_params[1] * control_input
        )
        self.state += dt * state_dot
        
        return self.state

def simulate_adaptive_control():
    """
    Simulate adaptive control system
    """
    adaptive_sys = AdaptiveRoboticSystem()
    
    # Simulation parameters
    dt = 0.01
    t_max = 10.0
    t = np.arange(0, t_max, dt)
    
    # Reference input (step command)
    reference = np.ones_like(t) * 1.0
    reference[int(0.5/dt):] = 2.0  # Step to 2.0 after 0.5 seconds
    
    # Arrays to store results
    states = []
    errors = []
    
    # Simulate
    for i in range(len(t)):
        state = adaptive_sys.step(reference[i], dt)
        states.append(state)
        errors.append(adaptive_sys.mrac.error)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, reference, 'r--', label='Reference', linewidth=2)
    plt.plot(t, states, 'b-', label='System Output', linewidth=2)
    plt.title('Adaptive Control Response')
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, errors, 'g-', label='Tracking Error', linewidth=2)
    plt.title('Tracking Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return t, reference, states, errors

# Example usage
if __name__ == "__main__":
    results = simulate_adaptive_control()
```

## Validation and Safety Considerations

Before deploying simulation-trained behaviors on real robots, thorough validation is essential.

### Safety-First Deployment Framework

```python
import numpy as np
from enum import Enum

class SafetyLevel(Enum):
    OFF = 0
    MONITORING = 1
    LIMITED = 2
    FULL = 3

class SafetyValidator:
    def __init__(self, safety_level=SafetyLevel.MONITORING):
        self.safety_level = safety_level
        self.safety_limits = {
            'position': 5.0,    # meters
            'velocity': 2.0,    # m/s
            'acceleration': 5.0, # m/s^2
            'torque': 100.0     # N-m for joints
        }
        self.emergency_stop = False
        
    def validate_action(self, action, current_state):
        """
        Validate an action before execution based on safety constraints
        """
        if self.emergency_stop:
            return False, "Emergency stop active"
        
        # Check if we're in a safe state
        is_safe_state, state_desc = self._check_safe_state(current_state)
        if not is_safe_state:
            return False, f"Unsafe state: {state_desc}"
        
        # Project the next state based on action
        projected_state = self._project_state(current_state, action)
        
        # Check if projected state is safe
        is_safe_action, action_desc = self._check_safe_action(projected_state)
        if not is_safe_action:
            return False, f"Unsafe action: {action_desc}"
        
        return True, "Action validated"
    
    def _check_safe_state(self, state):
        """
        Check if current state is safe for continued operation
        """
        # Check position limits
        if np.any(np.abs(state['position']) > self.safety_limits['position']):
            return False, "Position limit exceeded"
        
        # Check velocity limits
        if np.any(np.abs(state['velocity']) > self.safety_limits['velocity']):
            return False, "Velocity limit exceeded"
        
        # More checks can be added as needed
        return True, "State is safe"
    
    def _project_state(self, current_state, action, dt=0.01):
        """
        Project the next state based on action (simplified dynamics)
        """
        # This is a simplified state projection
        # In practice, this would use more detailed dynamics model
        projected_state = current_state.copy()
        
        # Update positions based on velocities
        projected_state['position'] += current_state['velocity'] * dt
        
        # Update velocities based on accelerations (estimated from action)
        estimated_acceleration = action * 0.1  # Simplified relationship
        projected_state['velocity'] += estimated_acceleration * dt
        
        return projected_state
    
    def _check_safe_action(self, projected_state):
        """
        Check if projected state is within safety limits
        """
        # Check position limits
        if np.any(np.abs(projected_state['position']) > self.safety_limits['position']):
            return False, "Projected position limit exceeded"
        
        # Check velocity limits
        if np.any(np.abs(projected_state['velocity']) > self.safety_limits['velocity']):
            return False, "Projected velocity limit exceeded"
        
        # More checks can be added as needed
        return True, "Action is safe"
    
    def trigger_emergency_stop(self, reason="Safety limit violation"):
        """
        Trigger emergency stop
        """
        print(f"EMERGENCY STOP: {reason}")
        self.emergency_stop = True
        
    def reset_emergency_stop(self):
        """
        Reset emergency stop
        """
        self.emergency_stop = False
        print("Emergency stop reset")

class SimToRealDeploymentManager:
    def __init__(self):
        self.safety_validator = SafetyValidator(SafetyLevel.FULL)
        self.confidence_threshold = 0.7  # Minimum confidence for action execution
        self.performance_threshold = 0.6  # Minimum performance for continued operation
        
    def deploy_with_validation(self, policy, state_estimator, robot_interface):
        """
        Deploy policy with safety and validation checks
        """
        print("Starting sim-to-real deployment with validation...")
        
        while True:
            # Get current state from robot
            current_state = state_estimator.get_state()
            
            # Get action from policy
            action, confidence = policy.get_action(current_state)
            
            # Validate action safety
            is_safe, reason = self.safety_validator.validate_action(action, current_state)
            
            if is_safe and confidence > self.confidence_threshold:
                # Execute action
                robot_interface.execute_action(action)
                print(f"Action executed with confidence: {confidence:.3f}")
            else:
                if not is_safe:
                    print(f"Action rejected due to safety: {reason}")
                    self.safety_validator.trigger_emergency_stop(reason)
                else:
                    print(f"Action rejected due to low confidence: {confidence:.3f}")
                
                # Execute safe action instead (e.g., stop)
                robot_interface.execute_safe_action()
            
            # Check overall system performance
            performance = robot_interface.get_performance_metric()
            if performance < self.performance_threshold:
                print(f"Performance below threshold: {performance:.3f}")
                self.safety_validator.trigger_emergency_stop("Low performance")
                break
                
            # Check for termination conditions
            if robot_interface.is_task_complete():
                print("Task completed successfully")
                break

# Example usage
def example_deployment():
    """
    Example of sim-to-real deployment with safety validation
    """
    class MockPolicy:
        def get_action(self, state):
            # Return a random action with some confidence
            action = np.random.uniform(-1, 1, size=state['position'].shape)
            confidence = np.random.uniform(0.5, 1.0)
            return action, confidence
    
    class MockStateEstimator:
        def get_state(self):
            return {
                'position': np.random.uniform(-1, 1, size=3),
                'velocity': np.random.uniform(-0.5, 0.5, size=3)
            }
    
    class MockRobotInterface:
        def execute_action(self, action):
            print(f"Executing action: {action[:2]}...")  # Show first 2 elements
            
        def execute_safe_action(self):
            print("Executing safe action (stop)")
            
        def get_performance_metric(self):
            return np.random.uniform(0.4, 1.0)
            
        def is_task_complete(self):
            # Randomly terminate after some time
            return np.random.rand() < 0.02  # ~2% chance per step
    
    # Create deployment manager
    deployment_manager = SimToRealDeploymentManager()
    
    # Create mock components
    policy = MockPolicy()
    state_estimator = MockStateEstimator()
    robot_interface = MockRobotInterface()
    
    # Deploy with validation
    deployment_manager.deploy_with_validation(policy, state_estimator, robot_interface)

if __name__ == "__main__":
    example_deployment()
```

## Exercise: Implement Domain Randomization for a Simple Robot

Create a simulation environment with domain randomization for a simple wheeled robot that needs to navigate to a target. Implement the randomization of friction, motor dynamics, and sensor noise, then train a policy that can adapt to these variations.

### Solution

```python
import numpy as np
import random

class SimpleWheeledRobotEnv:
    def __init__(self):
        # Robot state
        self.position = np.array([0.0, 0.0])  # x, y
        self.velocity = np.array([0.0, 0.0])  # dx, dy
        self.target = np.array([5.0, 5.0])    # Target position
        
        # Randomized parameters
        self.params = {
            'friction_coeff': 0.1,
            'motor_constant': 1.0,
            'sensor_noise_std': 0.01
        }
        
        # Action space: left_wheel_force, right_wheel_force
        self.action_space = 2
        # Observation space: x, y, dx, dy, target_x, target_y
        self.observation_space = 6
        
    def reset(self):
        """Reset environment with new randomization"""
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        
        # Randomize parameters for this episode
        self.params['friction_coeff'] = random.uniform(0.05, 0.3)
        self.params['motor_constant'] = random.uniform(0.8, 1.2)
        self.params['sensor_noise_std'] = random.uniform(0.005, 0.02)
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get observation with noise"""
        noise = np.random.normal(0, self.params['sensor_noise_std'], size=6)
        obs = np.concatenate([
            self.position,
            self.velocity,
            self.target
        ])
        return obs + noise
    
    def step(self, action):
        """Execute action and return (observation, reward, done, info)"""
        # Simplified 2-wheel robot dynamics
        left_force, right_force = action
        linear_force = (left_force + right_force) / 2
        angular_force = (right_force - left_force) / 2
        
        # Convert forces to x, y forces (simplified for this example)
        direction = np.arctan2(self.velocity[1], self.velocity[0])
        if np.linalg.norm(self.velocity) < 0.01:  # If not moving, use last direction
            direction = 0.0  # Default direction
            
        fx = linear_force * np.cos(direction) - angular_force * np.sin(direction)
        fy = linear_force * np.sin(direction) + angular_force * np.cos(direction)
        
        # Apply friction
        friction_force = -self.params['friction_coeff'] * self.velocity
        
        # Calculate acceleration (F = ma, assuming m=1)
        acceleration = np.array([fx, fy]) + friction_force
        
        # Update velocity and position
        dt = 0.1  # Time step
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Calculate reward (negative distance to target)
        distance_to_target = np.linalg.norm(self.position - self.target)
        reward = -distance_to_target
        
        # Check if done (reached target or too far)
        done = distance_to_target < 0.5 or np.linalg.norm(self.position) > 10.0
        
        return self._get_observation(), reward, done, {}
    
    def get_randomization_info(self):
        """Get current randomization parameters"""
        return self.params

def simple_training_loop():
    """Simple training loop with domain randomization"""
    env = SimpleWheeledRobotEnv()
    episodes = 1000
    
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        # Simple random policy for this example
        while steps < 100:
            # Generate random action (in a real implementation, 
            # this would come from a trained policy)
            action = np.random.uniform(-1.0, 1.0, size=2)
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            randomization_info = env.get_randomization_info()
            print(f"Episode {episode}: Total reward = {total_reward:.2f}")
            print(f"  Randomization: {randomization_info}")
    
    print("Training completed!")

if __name__ == "__main__":
    simple_training_loop()
```

## Takeaways

- The reality gap is a fundamental challenge in robotics that requires systematic approaches to address
- Domain randomization is a powerful technique that exposes the robot to a wide variety of conditions during training
- System identification helps quantify differences between simulation and reality
- Adaptive control systems can adjust their behavior to compensate for uncertainties
- Safety validation is essential before deploying simulation-trained behaviors on real robots
- Combining multiple techniques often yields better sim-to-real transfer than relying on a single approach

## Reading

- "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" by Tobin et al.
- "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization" by Peng et al.
- "System Identification: Theory for the User" by Ljung
- NVIDIA Isaac Lab Documentation on Domain Randomization

## Next Steps

Continue to Chapter 4: Isaac Simulation Overview to learn about the technical implementation of simulation environments in the Isaac ecosystem.