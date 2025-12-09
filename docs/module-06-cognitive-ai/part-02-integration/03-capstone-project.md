---
sidebar_position: 6
title: Capstone Project
---

# Capstone Project: Conversational Humanoid Robot

This capstone project integrates all the concepts learned throughout the textbook to create a conversational humanoid robot capable of understanding natural language, navigating environments, manipulating objects, and interacting naturally with humans. This comprehensive project demonstrates the full pipeline of Physical AI development from perception to action.

## Learning Objectives

- Integrate all modules learned in the textbook into a complete system
- Implement a conversational humanoid robot with multiple capabilities
- Apply best practices for system integration and validation
- Evaluate the performance of the complete system
- Document lessons learned from the integration process

## Introduction: The Complete Physical AI System

The capstone project brings together all components covered in the textbook to create a functional conversational humanoid robot. This system integrates:

- **ROS 2 Communication**: Coordinating all robot components
- **Simulation Environments**: Testing in Gazebo and NVIDIA Isaac Sim  
- **NVIDIA Isaac Framework**: Advanced perception and manipulation
- **Multimodal AI**: Vision, language, and audio processing
- **Conversational AI**: Natural language understanding and generation
- **Humanoid Control**: Locomotion and interaction capabilities

The project will demonstrate a robot that can:
- Understand and respond to natural language commands
- Navigate to specified locations
- Manipulate objects in its environment
- Maintain natural conversations with humans
- Integrate perception and action for robust operation

## Core System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Human-User Interface                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  Speech     │  │  Gesture    │  │  Touch/     │  │  Visual │ │
│  │ Recognition │  │ Recognition │  │ Haptics     │  │  UI     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                    ┌─────────────────────────┐
                    │     Dialogue Manager    │
                    │  ┌──────────────────┐   │
                    │  │  NLP Processor   │   │
                    │  └──────────────────┘   │
                    └─────────────────────────┘
                                   │
        ┌──────────────────────────────────────────────────────────┐
        │                 Task Planner                             │
        │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
        │  │ Navigation  │  │ Manipulation│  │ Interaction   │  │
        │  │   Tasks     │  │   Tasks     │  │   Manager     │  │
        │  └─────────────┘  └─────────────┘  └───────────────┘  │
        └──────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────────────────────────────────────┐
        │                ROS 2 Middleware                          │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐   │
        │  │ Vision   │  │ Motion   │  │ Control  │  │ Safety  │   │
        │  │ System   │  │ Planner  │  │ System   │  │System │   │
        │  └──────────┘  └──────────┘  └──────────┘  └─────────┘   │
        └──────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────────────────────────────────────┐
        │              Physical Robot Hardware                     │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐   │
        │  │ Sensors  │  │ Actuators│  │  CPUs/   │  │  Power  │   │
        │  │ & Perception││ & Effectors││  GPUs    │  │ System  │   │
        │  └──────────┘  └──────────┘  └──────────┘  └─────────┘   │
        └──────────────────────────────────────────────────────────┘
```

### System Components

**Perception Layer**: Processes visual, auditory, and other sensory inputs
**Cognition Layer**: Natural language understanding, reasoning, and planning
**Action Layer**: Navigation, manipulation, and physical interaction
**Interface Layer**: Natural human-robot interaction modalities

## Implementation Strategy

### 1. System Setup and Integration

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, LaserScan
from builtin_interfaces.msg import Duration
import cv2
import numpy as np
import threading
import queue
from typing import Dict, List, Optional
import time

class ConversationalHumanoidNode(Node):
    def __init__(self):
        super().__init__('conversational_humanoid')
        
        # Publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_publisher = self.create_publisher(String, '/speech_output', 10)
        
        # Subscribers for sensor data
        self.image_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.speech_subscriber = self.create_subscription(
            String, '/speech_input', self.speech_callback, 10
        )
        
        # Initialize components
        self.perception_system = PerceptionSystem()
        self.dialogue_manager = DialogueManager()
        self.task_planner = TaskPlanner()
        self.motion_controller = MotionController(self)
        
        # State management
        self.robot_state = {
            'location': [0, 0, 0],
            'orientation': 0.0,
            'battery_level': 100,
            'carrying_object': None,
            'current_task': None,
            'conversation_active': False
        }
        
        # Processing queues
        self.perception_queue = queue.Queue()
        self.dialogue_queue = queue.Queue()
        self.action_queue = queue.Queue()
        
        # Threads for parallel processing
        self.perception_thread = threading.Thread(target=self.perception_loop)
        self.dialogue_thread = threading.Thread(target=self.dialogue_loop)
        self.action_thread = threading.Thread(target=self.action_loop)
        
        # Start threads
        self.perception_thread.start()
        self.dialogue_thread.start()
        self.action_thread.start()
        
        self.get_logger().info('Conversational Humanoid Node initialized')

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            image = self.ros_to_cv2(msg)
            self.perception_queue.put(('image', image))
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def scan_callback(self, msg):
        """Process incoming LIDAR scans"""
        # Process laser scan data
        scan_data = {
            'ranges': msg.ranges,
            'intensities': msg.intensities,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }
        self.perception_queue.put(('scan', scan_data))

    def speech_callback(self, msg):
        """Process incoming speech commands"""
        self.dialogue_queue.put(('speech', msg.data))

    def ros_to_cv2(self, ros_image):
        """Convert ROS image message to OpenCV format"""
        # Implementation depends on the image encoding
        # This is a simplified implementation
        np_arr = np.frombuffer(ros_image.data, np.uint8)
        image = np_arr.reshape((ros_image.height, ros_image.width, -1))
        return image

    def perception_loop(self):
        """Continuous processing of sensor data"""
        while rclpy.ok():
            try:
                # Process all items in the queue
                while not self.perception_queue.empty():
                    data_type, data = self.perception_queue.get()
                    
                    if data_type == 'image':
                        # Process visual perception
                        objects = self.perception_system.detect_objects(data)
                        spatial_context = self.perception_system.analyze_scene(objects)
                        
                        # Update robot state with perception
                        self.robot_state['detected_objects'] = objects
                        self.robot_state['spatial_context'] = spatial_context
                        
                    elif data_type == 'scan':
                        # Process LIDAR data
                        obstacles = self.perception_system.detect_obstacles(data)
                        self.robot_state['obstacles'] = obstacles
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                self.get_logger().error(f'Perception loop error: {e}')
                time.sleep(0.1)

    def dialogue_loop(self):
        """Continuous processing of dialogue"""
        while rclpy.ok():
            try:
                # Process all items in the queue
                while not self.dialogue_queue.empty():
                    data_type, data = self.dialogue_queue.get()
                    
                    if data_type == 'speech':
                        # Process natural language
                        nlu_result = self.dialogue_manager.process_language(data, self.robot_state)
                        
                        if nlu_result:
                            # Add to action queue
                            self.action_queue.put(nlu_result)
                
                time.sleep(0.01)
                
            except Exception as e:
                self.get_logger().error(f'Dialogue loop error: {e}')
                time.sleep(0.1)

    def action_loop(self):
        """Continuous processing of actions"""
        while rclpy.ok():
            try:
                # Process all items in the queue
                while not self.action_queue.empty():
                    action_request = self.action_queue.get()
                    
                    # Plan and execute action
                    task_plan = self.task_planner.generate_plan(action_request, self.robot_state)
                    
                    if task_plan:
                        success = self.motion_controller.execute_plan(task_plan)
                        
                        # Generate response
                        response = self.dialogue_manager.generate_response(
                            action_request, 
                            success,
                            self.robot_state
                        )
                        
                        # Output response
                        response_msg = String()
                        response_msg.data = response
                        self.speech_publisher.publish(response_msg)
                
                time.sleep(0.01)
                
            except Exception as e:
                self.get_logger().error(f'Action loop error: {e}')
                time.sleep(0.1)

class PerceptionSystem:
    def __init__(self):
        self.object_detector = self._load_object_detector()
        self.scene_analyzer = SceneAnalyzer()
    
    def _load_object_detector(self):
        """Load or initialize object detection model"""
        # In a real implementation, this would load YOLO, Detectron2, or similar
        # For this example, we'll simulate the detector
        return "simulated_detector"
    
    def detect_objects(self, image):
        """Detect objects in an image"""
        # Simulate object detection
        # In a real system, this would run a deep learning model
        detected_objects = [
            {
                'name': 'cup',
                'bbox': [100, 200, 150, 250],  # [x1, y1, x2, y2]
                'confidence': 0.92,
                'position_3d': [1.2, 0.5, 0.0]  # Estimated 3D position
            },
            {
                'name': 'book',
                'bbox': [300, 150, 380, 280],
                'confidence': 0.87,
                'position_3d': [2.1, -0.3, 0.0]
            }
        ]
        return detected_objects
    
    def analyze_scene(self, objects):
        """Analyze relationships between objects in scene"""
        # Create spatial context from detected objects
        spatial_context = {
            'object_relations': self._compute_object_relations(objects),
            'room_layout': self._infer_room_layout(objects),
            'grasp_points': self._identify_grasp_points(objects)
        }
        return spatial_context
    
    def _compute_object_relations(self, objects):
        """Compute spatial relationships between objects"""
        relations = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Calculate simple spatial relationship
                pos1 = obj1['position_3d']
                pos2 = obj2['position_3d']
                
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    relation = f"{obj1['name']} is near {obj2['name']}"
                elif dx > 0:
                    relation = f"{obj2['name']} is to the right of {obj1['name']}"
                elif dx < 0:
                    relation = f"{obj2['name']} is to the left of {obj1['name']}"
                else:
                    relation = f"{obj1['name']} and {obj2['name']} are aligned vertically"
                
                relations.append(relation)
        
        return relations
    
    def detect_obstacles(self, scan_data):
        """Detect obstacles from LIDAR scan"""
        # Process scan data to identify obstacles
        ranges = scan_data['ranges']
        obstacles = []
        
        for i, range_val in enumerate(ranges):
            if not np.isnan(range_val) and range_val < 0.5:  # 0.5m threshold
                angle = scan_data['angle_min'] + i * scan_data['angle_increment']
                obstacle = {
                    'distance': range_val,
                    'angle': angle,
                    'x': range_val * np.cos(angle),
                    'y': range_val * np.sin(angle)
                }
                obstacles.append(obstacle)
        
        return obstacles
    
    def _infer_room_layout(self, objects):
        """Infer room layout from object positions"""
        # Simplified room layout inference
        if len(objects) >= 2:
            # Estimate room boundaries based on object positions
            xs = [obj['position_3d'][0] for obj in objects]
            ys = [obj['position_3d'][1] for obj in objects]
            
            layout = {
                'center': [np.mean(xs), np.mean(ys)],
                'bounds': {
                    'x_min': min(xs) - 1,
                    'x_max': max(xs) + 1,
                    'y_min': min(ys) - 1,
                    'y_max': max(ys) + 1
                }
            }
            return layout
        else:
            return {'center': [0, 0], 'bounds': {'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2}}
    
    def _identify_grasp_points(self, objects):
        """Identify potential grasp points for objects"""
        grasp_points = []
        for obj in objects:
            # For simple objects, approximate grasp point as center
            grasp_point = {
                'object': obj['name'],
                'position': obj['position_3d'],
                'approach_direction': [0, 0, -1]  # Approach from above
            }
            grasp_points.append(grasp_point)
        
        return grasp_points

class DialogueManager:
    def __init__(self):
        self.nlu = NaturalLanguageUnderstanding()
        self.response_generator = ResponseGenerator()
    
    def process_language(self, text, robot_state):
        """Process natural language and generate response"""
        # Parse the user's request
        nlu_result = self.nlu.process(text)
        
        if nlu_result.intent.value != 'unknown':
            # Return the parsed action request
            return {
                'intent': nlu_result.intent.value,
                'entities': {e.type: e.value for e in nlu_result.entities},
                'confidence': nlu_result.confidence,
                'original_text': text
            }
        
        return None
    
    def generate_response(self, action_request, success, robot_state):
        """Generate a natural language response"""
        if success:
            if action_request['intent'] == 'command_move':
                return f"I've moved to the {action_request['entities'].get('location', 'location')}."
            elif action_request['intent'] == 'command_manipulate':
                return f"I've picked up the {action_request['entities'].get('object', 'object')}."
            else:
                return "I've completed the requested task."
        else:
            return "I couldn't complete that task. Could you please try again?"

class TaskPlanner:
    def __init__(self):
        self.action_sequences = {
            'navigate': self._plan_navigation,
            'grasp': self._plan_grasping,
            'transport': self._plan_transport,
            'answer_query': self._plan_query_response
        }
    
    def generate_plan(self, action_request, robot_state):
        """Generate a plan for the requested action"""
        intent = action_request['intent']
        
        if intent in self.action_sequences:
            return self.action_sequences[intent](action_request, robot_state)
        else:
            return self._plan_generic(action_request, robot_state)
    
    def _plan_navigation(self, action_request, robot_state):
        """Plan navigation to a location"""
        target_location = action_request['entities'].get('location')
        
        if not target_location:
            return None
        
        plan = [
            {
                'action': 'navigate',
                'parameters': {'target_location': target_location},
                'description': f'Navigating to {target_location}',
                'estimated_time': 30  # seconds
            }
        ]
        return plan
    
    def _plan_grasping(self, action_request, robot_state):
        """Plan grasping of an object"""
        target_object = action_request['entities'].get('object')
        
        if not target_object:
            return None
        
        # Find object in robot's perception
        detected_objects = robot_state.get('detected_objects', [])
        target_obj_info = None
        
        for obj in detected_objects:
            if target_object.lower() in obj['name'].lower():
                target_obj_info = obj
                break
        
        if not target_obj_info:
            # Object not found, may need to navigate to search
            return [
                {
                    'action': 'search',
                    'parameters': {'object': target_object},
                    'description': f'Searching for {target_object}',
                    'estimated_time': 60
                }
            ]
        
        plan = [
            {
                'action': 'approach',
                'parameters': {'object_location': target_obj_info['position_3d']},
                'description': f'Approaching {target_object}',
                'estimated_time': 15
            },
            {
                'action': 'grasp',
                'parameters': {'object': target_object},
                'description': f'Grasping {target_object}',
                'estimated_time': 10
            }
        ]
        return plan
    
    def _plan_transport(self, action_request, robot_state):
        """Plan transporting an object to a location"""
        target_object = action_request['entities'].get('object')
        target_location = action_request['entities'].get('location')
        
        if not target_object or not target_location:
            return None
        
        plan = [
            {
                'action': 'grasp',
                'parameters': {'object': target_object},
                'description': f'Grasping {target_object}',
                'estimated_time': 10
            },
            {
                'action': 'navigate',
                'parameters': {'target_location': target_location},
                'description': f'Navigating to {target_location} with {target_object}',
                'estimated_time': 45
            },
            {
                'action': 'place',
                'parameters': {'object': target_object, 'location': target_location},
                'description': f'Placing {target_object} at {target_location}',
                'estimated_time': 10
            }
        ]
        return plan
    
    def _plan_query_response(self, action_request, robot_state):
        """Plan response to query"""
        query_type = action_request['entities'].get('query_type', 'general')
        
        plan = [
            {
                'action': 'formulate_response',
                'parameters': {'query_type': query_type, 'context': robot_state},
                'description': f'Formulating response to {query_type} query',
                'estimated_time': 5
            }
        ]
        return plan
    
    def _plan_generic(self, action_request, robot_state):
        """Default plan for unrecognized intents"""
        return [
            {
                'action': 'ask_for_clarification',
                'parameters': {'original_request': action_request['original_text']},
                'description': 'Asking for clarification',
                'estimated_time': 5
            }
        ]
```

### 2. Motion Control and Navigation

```python
class MotionController:
    def __init__(self, node):
        self.node = node
        self.nav_client = self._create_navigation_client()
        self.arm_client = self._create_manipulation_client()
        self.current_plan = None
    
    def _create_navigation_client(self):
        """Create navigation client for path planning and execution"""
        # In a real implementation, this would connect to Nav2
        # For this example, we'll simulate the navigation client
        return "simulated_nav_client"
    
    def _create_manipulation_client(self):
        """Create manipulation client for arm control"""
        # In a real implementation, this would connect to MoveIt2 or similar
        # For this example, we'll simulate the manipulation client
        return "simulated_manipulation_client"
    
    def execute_plan(self, plan):
        """Execute a task plan step by step"""
        success = True
        
        for step in plan:
            action = step['action']
            params = step['parameters']
            
            self.node.get_logger().info(f"Executing: {step['description']}")
            
            # Execute the action
            if action == 'navigate':
                success = self._execute_navigation(params)
            elif action == 'approach':
                success = self._execute_approach(params)
            elif action == 'grasp':
                success = self._execute_grasping(params)
            elif action == 'place':
                success = self._execute_placement(params)
            elif action == 'search':
                success = self._execute_search(params)
            else:
                # For simulated actions, just delay to simulate execution time
                time.sleep(step['estimated_time'])
                success = True
            
            # If any step fails, the whole plan fails
            if not success:
                self.node.get_logger().error(f"Action failed: {step['description']}")
                break
        
        return success
    
    def _execute_navigation(self, params):
        """Execute navigation to a location"""
        target_location = params.get('target_location', [0, 0, 0])
        
        # Simulate navigation execution
        self.node.get_logger().info(f"Navigating to {target_location}")
        
        # In a real system, this would send navigation goals to Nav2
        # Simulate navigation time
        time.sleep(5)  # Simulated navigation time
        
        # Update robot state
        if hasattr(self.node, 'robot_state'):
            self.node.robot_state['location'] = target_location
        
        return True  # Simulated success
    
    def _execute_approach(self, params):
        """Execute approach to an object"""
        object_location = params.get('object_location', [0, 0, 0])
        
        # Simulate approach execution
        self.node.get_logger().info(f"Approaching object at {object_location}")
        
        # Simulate approach time
        time.sleep(3)
        
        return True  # Simulated success
    
    def _execute_grasping(self, params):
        """Execute grasping of an object"""
        target_object = params.get('object', 'unknown object')
        
        # Simulate grasping execution
        self.node.get_logger().info(f"Grasping {target_object}")
        
        # Simulate grasping time
        time.sleep(4)
        
        # Update robot state
        if hasattr(self.node, 'robot_state'):
            self.node.robot_state['carrying_object'] = target_object
        
        return True  # Simulated success
    
    def _execute_placement(self, params):
        """Execute placement of an object"""
        target_location = params.get('location', [0, 0, 0])
        target_object = params.get('object', 'unknown object')
        
        # Simulate placement execution
        self.node.get_logger().info(f"Placing {target_object} at {target_location}")
        
        # Simulate placement time
        time.sleep(3)
        
        # Update robot state
        if hasattr(self.node, 'robot_state'):
            self.node.robot_state['carrying_object'] = None
        
        return True  # Simulated success
    
    def _execute_search(self, params):
        """Execute search for an object"""
        target_object = params.get('object', 'unknown object')
        
        # Simulate search execution
        self.node.get_logger().info(f"Searching for {target_object}")
        
        # Simulate search time
        time.sleep(30)
        
        # For simulation, assume search is successful
        # In a real system, this would involve actual perception and navigation
        return True  # Simulated success

class SceneAnalyzer:
    def __init__(self):
        self.navigation_map = {}
        self.object_affordances = {}
    
    def analyze_for_navigation(self, spatial_context):
        """Analyze scene for safe navigation paths"""
        # Identify navigable areas and obstacles
        room_bounds = spatial_context.get('room_layout', {}).get('bounds', {})
        
        # Create a simple grid-based representation for navigation planning
        grid_size = 0.1  # 10cm resolution
        x_range = np.arange(room_bounds.get('x_min', -2), room_bounds.get('x_max', 2), grid_size)
        y_range = np.arange(room_bounds.get('y_min', -2), room_bounds.get('y_max', 2), grid_size)
        
        # Mark occupied cells based on obstacles
        occupancy_grid = np.zeros((len(y_range), len(x_range)))
        
        obstacles = spatial_context.get('obstacles', [])
        for obstacle in obstacles:
            x_idx = int((obstacle['x'] - room_bounds.get('x_min', -2)) / grid_size)
            y_idx = int((obstacle['y'] - room_bounds.get('y_min', -2)) / grid_size)
            
            if 0 <= x_idx < len(x_range) and 0 <= y_idx < len(y_range):
                occupancy_grid[y_idx, x_idx] = 1  # Mark as occupied
        
        return {
            'grid': occupancy_grid,
            'resolution': grid_size,
            'bounds': room_bounds
        }
    
    def analyze_for_manipulation(self, detected_objects):
        """Analyze scene for manipulation opportunities"""
        manipulation_targets = []
        
        for obj in detected_objects:
            obj_name = obj['name']
            obj_pos = obj['position_3d']
            
            # Determine manipulation affordances
            affordances = self._get_object_affordances(obj_name)
            
            manipulation_targets.append({
                'object': obj_name,
                'position': obj_pos,
                'affordances': affordances,
                'grasp_points': self._compute_grasp_points(obj)
            })
        
        return manipulation_targets
    
    def _get_object_affordances(self, obj_name):
        """Get possible actions for an object"""
        affordances = {
            'cup': ['grasp', 'lift', 'carry', 'place'],
            'book': ['grasp', 'lift', 'carry', 'place', 'open'],
            'bottle': ['grasp', 'lift', 'carry', 'place', 'pour'],
            'box': ['grasp', 'lift', 'carry', 'place'],
            'chair': ['move', 'reposition'],
            'door': ['open', 'close']
        }
        
        return affordances.get(obj_name.lower(), ['grasp', 'move'])
    
    def _compute_grasp_points(self, obj):
        """Compute possible grasp points for an object"""
        # For a simple object like a cup, the grasp points might be:
        # - Handle (if present)
        # - Sides of the cup
        # - Top rim (for special grippers)
        
        obj_name = obj['name'].lower()
        center_pos = obj['position_3d']
        
        grasp_points = [
            {
                'position': [center_pos[0], center_pos[1], center_pos[2] + 0.1],  # Above center
                'approach_direction': [0, 0, -1],  # From above
                'grip_type': 'top_grasp',
                'confidence': 0.9
            }
        ]
        
        if 'cup' in obj_name:
            grasp_points.append({
                'position': [center_pos[0] + 0.05, center_pos[1], center_pos[2]],  # Side of cup
                'approach_direction': [-1, 0, 0],  # From the side
                'grip_type': 'side_grasp',
                'confidence': 0.8
            })
        
        return grasp_points
```

### 3. Natural Language Integration

```python
from enum import Enum
from dataclasses import dataclass

class IntentType(Enum):
    GREETING = "greeting"
    COMMAND_MOVE = "command_move"
    COMMAND_MANIPULATE = "command_manipulate"
    INQUIRY_STATUS = "inquiry_status"
    INQUIRY_LOCATION = "inquiry_location"
    INQUIRY_CAPABILITY = "inquiry_capability"
    STOP = "stop"
    UNKNOWN = "unknown"

@dataclass
class NLUResult:
    intent: IntentType
    entities: List[object]  # We'll use a simple structure
    confidence: float
    original_text: str

class NaturalLanguageUnderstanding:
    def __init__(self):
        # Define patterns for different intents
        self.patterns = {
            IntentType.GREETING: [
                r"hello\b", r"hi\b", r"hey\b", r"greetings\b",
                r"good morning\b", r"good afternoon\b", r"good evening\b"
            ],
            IntentType.COMMAND_MOVE: [
                r"go to\b", r"move to\b", r"navigate to\b",
                r"walk to\b", r"get to\b", r"head to\b",
                r"bring me to\b", r"take me to\b"
            ],
            IntentType.COMMAND_MANIPULATE: [
                r"pick up\b", r"grasp\b", r"grab\b", r"take\b",
                r"lift\b", r"get\b", r"bring me\b", r"move\b"
            ],
            IntentType.INQUIRY_STATUS: [
                r"how are you\b", r"what's your status\b", r"are you okay\b",
                r"report status\b", r"what can you do\b", r"what are you doing\b"
            ],
            IntentType.INQUIRY_LOCATION: [
                r"where are you\b", r"where is\b", r"locate\b", r"find\b",
                r"search for\b", r"look for\b"
            ],
            IntentType.INQUIRY_CAPABILITY: [
                r"what can you do\b", r"what are your capabilities\b",
                r"what's possible\b", r"help\b", r"what are you able\b"
            ],
            IntentType.STOP: [
                r"stop\b", r"halt\b", r"pause\b", r"freeze\b", r"wait\b"
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'location': [
                r"\bto\s+([a-zA-Z\s]+?)(?:\s|$)",
                r"\b(at|in|on)\s+([a-zA-Z\s]+?)(?:\s|$)",
                r"\b(room|area|zone|spot)\s+([a-zA-Z\s]+?)(?:\s|$)",
            ],
            'object': [
                r"\b(pick up|grasp|take|grab|move|place|put|get)\s+([a-zA-Z\s]+?)(?:\s|$)",
                r"\b(the\s+)?([a-zA-Z\s]+?)\s+(on\s+the\s+table|on\s+the\s+floor|there|here)\b"
            ],
            'person': [
                r"\b(person|someone|you|me)\b"
            ]
        }
    
    def process(self, text: str) -> NLUResult:
        """Process natural language text and extract intent and entities"""
        text_lower = text.lower()
        
        # Identify intent
        intent, confidence = self._identify_intent(text_lower)
        
        # Extract entities
        entities = self._extract_entities(text_lower)
        
        return NLUResult(
            intent=intent,
            entities=entities,
            confidence=confidence,
            original_text=text
        )
    
    def _identify_intent(self, text_lower: str) -> tuple[IntentType, float]:
        """Identify the intent of the given text"""
        best_intent = IntentType.UNKNOWN
        best_score = 0
        
        for intent, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                import re
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score += len(matches)
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Calculate confidence based on match strength
        confidence = min(1.0, best_score * 0.3)  # Adjust scaling factor as needed
        return best_intent, confidence
    
    def _extract_entities(self, text_lower: str) -> List[object]:
        """Extract named entities from text"""
        import re
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Extract the relevant part of the match
                    value = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    if value and len(value.strip()) > 0:
                        entities.append(type('', (), {'type': entity_type, 'value': value.strip(), 'confidence': 0.8})())
        
        # Remove duplicate entities
        unique_entities = []
        for entity in entities:
            is_duplicate = False
            for unique_entity in unique_entities:
                if unique_entity.value == entity.value and unique_entity.type == entity.type:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities

class ResponseGenerator:
    def __init__(self):
        self.response_templates = {
            IntentType.GREETING: [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Greetings! I'm ready to help.",
                "Good day! How may I assist you?"
            ],
            IntentType.COMMAND_MOVE: [
                "I'll move to the {location} right away.",
                "On my way to the {location}.",
                "Navigating to the {location} now.",
                "I'm going to the {location} as requested."
            ],
            IntentType.COMMAND_MANIPULATE: [
                "I'll {action} the {object} for you.",
                "Picking up the {object} now.",
                "Manipulating the {object} as requested.",
                "I'm going to {action} the {object}."
            ],
            IntentType.INQUIRY_STATUS: [
                "I'm functioning normally and ready for tasks.",
                "All systems operational. I can assist with various tasks.",
                "I'm in good condition and ready to help.",
                "I'm ready to perform actions as needed."
            ],
            IntentType.INQUIRY_LOCATION: [
                "I'm currently located in the {location}.",
                "My position is in the {location} area.",
                "I'm situated in the {location}.",
                "I'm positioned in the {location}."
            ],
            IntentType.INQUIRY_CAPABILITY: [
                "I can perform tasks like navigation, object manipulation, and answering questions.",
                "My capabilities include moving to locations, grasping objects, and communicating.",
                "I'm able to navigate spaces, manipulate objects, and interact through speech."
            ],
            IntentType.STOP: [
                "Stopping all actions.",
                "All movement stopped.",
                "Halted all operations.",
                "Stopping as requested."
            ],
            IntentType.UNKNOWN: [
                "I'm not sure I understand. Could you rephrase that?",
                "I didn't catch that. Could you say it again?",
                "I'm not sure what you mean. Could you clarify?",
                "I don't recognize that command. Please try something else."
            ]
        }
    
    def generate_response(self, action_request, success, robot_state):
        """Generate a contextual response based on action and robot state"""
        import random
        
        # Get intent and entities
        intent_str = action_request.get('intent', 'unknown')
        entities = action_request.get('entities', {})
        
        # Get appropriate template
        if success:
            if intent_str == 'command_move':
                template = random.choice(self.response_templates.get(IntentType.COMMAND_MOVE, ["OK"]))
                response = template.format(location=entities.get('location', 'destination'))
            elif intent_str == 'command_manipulate':
                template = random.choice(self.response_templates.get(IntentType.COMMAND_MANIPULATE, ["OK"]))
                response = template.format(
                    action='grasp',  # Default action
                    object=entities.get('object', 'item')
                )
            elif intent_str == 'inquiry_status':
                response = random.choice(self.response_templates.get(IntentType.INQUIRY_STATUS, ["I'm ready"]))
            elif intent_str == 'inquiry_location':
                response = random.choice(self.response_templates.get(IntentType.INQUIRY_LOCATION, ["I'm here"]))
            elif intent_str == 'greeting':
                response = random.choice(self.response_templates.get(IntentType.GREETING, ["Hello"]))
            else:
                response = "Task completed successfully."
        else:
            response = "I couldn't complete that task. Could you please try again?"
        
        return response
```

### 4. System Integration and Testing

```python
def main():
    """Main function to run the conversational humanoid robot"""
    rclpy.init()
    
    # Create the node
    node = ConversationalHumanoidNode()
    
    try:
        # Run the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.get_logger().info('Shutting down Conversational Humanoid Node')
        if node.perception_thread.is_alive():
            node.perception_thread.join(timeout=1.0)
        if node.dialogue_thread.is_alive():
            node.dialogue_thread.join(timeout=1.0)
        if node.action_thread.is_alive():
            node.action_thread.join(timeout=1.0)
        
        node.destroy_node()
        rclpy.shutdown()

class SystemValidator:
    def __init__(self):
        self.test_results = {}
    
    def run_integration_tests(self):
        """Run comprehensive integration tests"""
        tests = [
            self._test_perception_pipeline,
            self._test_dialogue_understanding,
            self._test_task_planning,
            self._test_navigation_integration,
            self._test_manipulation_integration
        ]
        
        results = {}
        for test in tests:
            test_name = test.__name__
            try:
                success, details = test()
                results[test_name] = {'success': success, 'details': details}
            except Exception as e:
                results[test_name] = {'success': False, 'error': str(e)}
        
        self.test_results = results
        return results
    
    def _test_perception_pipeline(self):
        """Test the perception system with sample data"""
        # Create a sample image
        sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Initialize perception
        perception = PerceptionSystem()
        objects = perception.detect_objects(sample_image)
        
        # Validate results
        if objects:
            return True, f"Detected {len(objects)} objects"
        else:
            return False, "No objects detected"
    
    def _test_dialogue_understanding(self):
        """Test the dialogue understanding system"""
        dialogue_manager = DialogueManager()
        
        test_inputs = [
            "Go to the kitchen",
            "Pick up the red cup",
            "How are you?",
            "What can you do?"
        ]
        
        success_count = 0
        for input_text in test_inputs:
            result = dialogue_manager.process_language(input_text, {})
            if result and result['intent'] != 'unknown':
                success_count += 1
        
        success_rate = success_count / len(test_inputs)
        return success_rate >= 0.75, f"Successfully parsed {success_count}/{len(test_inputs)} inputs"
    
    def _test_task_planning(self):
        """Test the task planning system"""
        planner = TaskPlanner()
        
        test_requests = [
            {
                'intent': 'command_move',
                'entities': {'location': 'kitchen'},
                'confidence': 0.9
            },
            {
                'intent': 'command_manipulate',
                'entities': {'object': 'cup'},
                'confidence': 0.8
            }
        ]
        
        success_count = 0
        for request in test_requests:
            plan = planner.generate_plan(request, {})
            if plan:
                success_count += 1
        
        success_rate = success_count / len(test_requests)
        return success_rate >= 0.5, f"Successfully planned {success_count}/{len(test_requests)} tasks"
    
    def _test_navigation_integration(self):
        """Test navigation system integration"""
        # This would test actual navigation in simulation
        # For this example, we'll simulate success
        return True, "Navigation system integrated successfully"
    
    def _test_manipulation_integration(self):
        """Test manipulation system integration"""
        # This would test actual manipulation in simulation
        # For this example, we'll simulate success
        return True, "Manipulation system integrated successfully"
    
    def generate_system_report(self):
        """Generate a comprehensive system validation report"""
        if not self.test_results:
            self.run_integration_tests()
        
        report = {
            'timestamp': time.ctime(),
            'total_tests': len(self.test_results),
            'successful_tests': sum(1 for r in self.test_results.values() if r['success']),
            'test_results': self.test_results,
            'system_status': 'operational' if all(r['success'] for r in self.test_results.values()) else 'issues_found'
        }
        
        return report

# Example usage of system validator
if __name__ == '__main__':
    # For the actual robot node, we would run main()
    # For testing purposes, let's run the system validation
    
    validator = SystemValidator()
    validation_report = validator.generate_system_report()
    
    print("=== System Validation Report ===")
    print(f"Timestamp: {validation_report['timestamp']}")
    print(f"System Status: {validation_report['system_status']}")
    print(f"Tests Passed: {validation_report['successful_tests']}/{validation_report['total_tests']}")
    print("\nDetailed Results:")
    for test_name, result in validation_report['test_results'].items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"  {test_name}: {status} - {result.get('details', result.get('error', 'Unknown'))}")
    
    # Only run the full system if we're executing the main script
    # Commenting out the main() call for this example
    # main()
```

## System Performance Evaluation

### Evaluation Metrics

```python
class SystemEvaluator:
    def __init__(self):
        self.metrics = {
            'response_accuracy': 0.0,
            'task_completion_rate': 0.0,
            'navigation_success_rate': 0.0,
            'object_manipulation_success_rate': 0.0,
            'dialogue_coherence': 0.0,
            'system_latency': 0.0,
            'user_satisfaction': 0.0
        }
    
    def evaluate_response_accuracy(self, test_cases):
        """Evaluate the accuracy of system responses"""
        correct_responses = 0
        total_cases = len(test_cases)
        
        for test_input, expected_output in test_cases:
            # Simulate getting actual output from the system
            actual_output = self._simulate_system_response(test_input)
            
            if self._compare_responses(actual_output, expected_output):
                correct_responses += 1
        
        accuracy = correct_responses / total_cases if total_cases > 0 else 0
        self.metrics['response_accuracy'] = accuracy
        return accuracy
    
    def evaluate_task_completion(self, task_list):
        """Evaluate task completion rates"""
        completed_tasks = 0
        total_tasks = len(task_list)
        
        for task in task_list:
            success = self._simulate_task_execution(task)
            if success:
                completed_tasks += 1
        
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        self.metrics['task_completion_rate'] = completion_rate
        return completion_rate
    
    def _simulate_system_response(self, input_text):
        """Simulate getting a response from the system"""
        # This would normally query the live system
        # For simulation, return a placeholder
        if "hello" in input_text.lower():
            return "Hello! How can I help you?"
        elif "go to" in input_text.lower():
            return "I'm on my way."
        elif "pick up" in input_text.lower():
            return "I'll pick that up for you."
        else:
            return "I received your request."
    
    def _compare_responses(self, actual, expected):
        """Compare two responses for similarity"""
        # Simplified comparison - in practice, you'd use more sophisticated methods
        return actual.lower() == expected.lower()
    
    def _simulate_task_execution(self, task):
        """Simulate executing a task"""
        # Simulate based on task type
        import random
        return random.random() > 0.2  # 80% success rate for simulation
    
    def run_comprehensive_evaluation(self):
        """Run all evaluation metrics"""
        # Define test cases
        test_cases = [
            ("Hello robot", "Hello! How can I help you?"),
            ("Go to the kitchen", "Navigating to kitchen"),
            ("Pick up the red cup", "Picking up red cup")
        ]
        
        tasks = [
            {'type': 'navigation', 'destination': 'kitchen'},
            {'type': 'manipulation', 'object': 'cup'},
            {'type': 'greeting'}
        ]
        
        # Run evaluations
        response_accuracy = self.evaluate_response_accuracy(test_cases)
        task_completion_rate = self.evaluate_task_completion(tasks)
        
        # Update metrics
        self.metrics['response_accuracy'] = response_accuracy
        self.metrics['task_completion_rate'] = task_completion_rate
        
        return self.metrics
    
    def generate_evaluation_report(self):
        """Generate a comprehensive evaluation report"""
        metrics = self.run_comprehensive_evaluation()
        
        report = f"""
        Conversational Humanoid Robot - System Evaluation Report
        ========================================================
        
        Date: {time.ctime()}
        System Version: 1.0
        
        Metrics:
        - Response Accuracy: {metrics['response_accuracy']:.2%}
        - Task Completion Rate: {metrics['task_completion_rate']:.2%}
        - Navigation Success Rate: {metrics['navigation_success_rate']:.2%}
        - Object Manipulation Success Rate: {metrics['object_manipulation_success_rate']:.2%}
        - Dialogue Coherence: {metrics['dialogue_coherence']:.2%}
        - Average System Latency: {metrics['system_latency']:.2f}s
        - User Satisfaction Score: {metrics['user_satisfaction']:.2f}/10
        
        Summary:
        The system demonstrates a{'n' if metrics['response_accuracy'] > 0.7 else ''} 
        {'effective' if metrics['response_accuracy'] > 0.7 else 'improvement-needed'} 
        performance with a response accuracy of {metrics['response_accuracy']:.2%}.
        
        Recommendations:
        - {'Continue current development approach' if metrics['response_accuracy'] > 0.7 else 'Focus on improving natural language understanding'}
        - {'Expand testing scenarios' if metrics['task_completion_rate'] > 0.8 else 'Improve task planning and execution reliability'}
        """
        
        return report

# Example evaluation usage
if __name__ == "__main__":
    evaluator = SystemEvaluator()
    report = evaluator.generate_evaluation_report()
    print(report)
```

## Deployment Considerations

### Configuration Management

```python
import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RobotConfiguration:
    """Configuration for the conversational humanoid robot"""
    robot_name: str = "conversational_humanoid"
    robot_model: str = "custom_humanoid"
    
    # Perception settings
    camera_topic: str = "/camera/image_raw"
    lidar_topic: str = "/scan"
    audio_input_topic: str = "/audio_input"
    audio_output_topic: str = "/audio_output"
    
    # Performance settings
    control_frequency: float = 10.0  # Hz
    max_navigation_speed: float = 0.5  # m/s
    min_approach_distance: float = 0.3  # m
    
    # AI model paths
    vision_model_path: str = "/models/vision_model.onnx"
    language_model_path: str = "/models/language_model.onnx"
    
    # Safety settings
    safety_distance_threshold: float = 0.5  # m
    maximum_operating_time: int = 3600  # seconds
    emergency_stop_timeout: float = 5.0  # seconds
    
    def save_to_file(self, filepath: str):
        """Save configuration to YAML file"""
        config_dict = {
            'robot_name': self.robot_name,
            'robot_model': self.robot_model,
            'perception': {
                'camera_topic': self.camera_topic,
                'lidar_topic': self.lidar_topic,
                'audio_input_topic': self.audio_input_topic,
                'audio_output_topic': self.audio_output_topic
            },
            'performance': {
                'control_frequency': self.control_frequency,
                'max_navigation_speed': self.max_navigation_speed,
                'min_approach_distance': self.min_approach_distance
            },
            'ai_models': {
                'vision_model_path': self.vision_model_path,
                'language_model_path': self.language_model_path
            },
            'safety': {
                'safety_distance_threshold': self.safety_distance_threshold,
                'maximum_operating_time': self.maximum_operating_time,
                'emergency_stop_timeout': self.emergency_stop_timeout
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create instance with default values first
        instance = cls()
        
        # Update with loaded values
        instance.robot_name = config_dict.get('robot_name', instance.robot_name)
        instance.robot_model = config_dict.get('robot_model', instance.robot_model)
        
        # Load perception settings
        perception = config_dict.get('perception', {})
        instance.camera_topic = perception.get('camera_topic', instance.camera_topic)
        instance.lidar_topic = perception.get('lidar_topic', instance.lidar_topic)
        instance.audio_input_topic = perception.get('audio_input_topic', instance.audio_input_topic)
        instance.audio_output_topic = perception.get('audio_output_topic', instance.audio_output_topic)
        
        # Load performance settings
        performance = config_dict.get('performance', {})
        instance.control_frequency = performance.get('control_frequency', instance.control_frequency)
        instance.max_navigation_speed = performance.get('max_navigation_speed', instance.max_navigation_speed)
        instance.min_approach_distance = performance.get('min_approach_distance', instance.min_approach_distance)
        
        # Load AI model paths
        ai_models = config_dict.get('ai_models', {})
        instance.vision_model_path = ai_models.get('vision_model_path', instance.vision_model_path)
        instance.language_model_path = ai_models.get('language_model_path', instance.language_model_path)
        
        # Load safety settings
        safety = config_dict.get('safety', {})
        instance.safety_distance_threshold = safety.get('safety_distance_threshold', instance.safety_distance_threshold)
        instance.maximum_operating_time = safety.get('maximum_operating_time', instance.maximum_operating_time)
        instance.emergency_stop_timeout = safety.get('emergency_stop_timeout', instance.emergency_stop_timeout)
        
        return instance

# Example usage
if __name__ == "__main__":
    # Create default configuration
    default_config = RobotConfiguration()
    
    # Save to file
    config_file = "conversational_humanoid_config.yaml"
    default_config.save_to_file(config_file)
    print(f"Configuration saved to {config_file}")
    
    # Load from file
    loaded_config = RobotConfiguration.load_from_file(config_file)
    print(f"Configuration loaded: {loaded_config.robot_name}")
```

## Troubleshooting and Maintenance

### Common Issues and Solutions

```python
class TroubleshootingGuide:
    def __init__(self):
        self.issues = {
            'navigation_failure': {
                'symptoms': ['robot stops moving', 'cannot reach destination', 'path planning errors'],
                'causes': ['obstacle in path', 'incorrect map', 'localization error'],
                'solutions': [
                    'Clear obstacles from path',
                    'Update map with current environment',
                    'Re-localize robot position'
                ]
            },
            'object_detection_failure': {
                'symptoms': ['cannot find requested object', 'low detection confidence', 'false positives'],
                'causes': ['poor lighting', 'occluded objects', 'model limitations'],
                'solutions': [
                    'Improve lighting conditions',
                    'Move robot for better view angle',
                    'Update detection models with new data'
                ]
            },
            'speech_recognition_failure': {
                'symptoms': ['commands not understood', 'high error rate', 'no response to speech'],
                'causes': ['background noise', 'microphone issues', 'model limitations'],
                'solutions': [
                    'Reduce background noise',
                    'Check microphone connections',
                    'Use noise reduction algorithms'
                ]
            },
            'manipulation_failure': {
                'symptoms': ['cannot grasp object', 'unstable grasp', 'dropped objects'],
                'causes': ['incorrect grasping point', 'object properties', 'suction issues'],
                'solutions': [
                    'Recalculate grasp points',
                    'Adjust gripper parameters',
                    'Verify object properties'
                ]
            }
        }
    
    def diagnose_issue(self, symptoms):
        """Diagnose issue based on symptoms"""
        matches = {}
        for issue_name, issue_info in self.issues.items():
            symptom_matches = [sym for sym in symptoms if any(sym.lower() in s.lower() for s in issue_info['symptoms'])]
            if symptom_matches:
                matches[issue_name] = {
                    'matched_symptoms': symptom_matches,
                    'confidence': len(symptom_matches) / len(issue_info['symptoms'])
                }
        
        return matches
    
    def get_solution(self, issue_name):
        """Get solutions for a specific issue"""
        if issue_name in self.issues:
            return self.issues[issue_name]['solutions']
        else:
            return ["No solution found for this issue"]

# Example usage
troubleshooter = TroubleshootingGuide()
issue_matches = troubleshooter.diagnose_issue(['robot stops moving', 'cannot reach destination'])
print(f"Potential issues: {issue_matches}")

if issue_matches:
    best_match = max(issue_matches.items(), key=lambda x: x[1]['confidence'])
    print(f"Most likely issue: {best_match[0]}")
    solutions = troubleshooter.get_solution(best_match[0])
    print(f"Suggested solutions: {solutions}")
```

## Deployment and Production Considerations

### System Monitoring

```python
import psutil
import GPUtil
from datetime import datetime
import json

class SystemMonitor:
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
    
    def collect_system_metrics(self):
        """Collect system resource metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'robot_status': 'active',  # Would be retrieved from robot state
            'components_active': 3,    # Number of active system components
            'errors_count': 0          # Would be retrieved from logs
        }
        
        # Get GPU usage if available
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            metrics['gpu_percent'] = gpu.load * 100
            metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
        else:
            metrics['gpu_percent'] = 0
            metrics['gpu_memory_percent'] = 0
        
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_for_alerts(metrics)
        
        return metrics
    
    def _check_for_alerts(self, metrics):
        """Check if any metrics exceed thresholds"""
        alerts = []
        
        if metrics['cpu_percent'] > 90:
            alerts.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        if metrics['memory_percent'] > 90:
            alerts.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
        
        if metrics['gpu_percent'] > 95:
            alerts.append(f"High GPU usage: {metrics['gpu_percent']:.1f}%")
        
        if metrics['gpu_memory_percent'] > 95:
            alerts.append(f"High GPU memory usage: {metrics['gpu_memory_percent']:.1f}%")
        
        if len(alerts) > 0:
            self.alerts.extend(alerts)
            print(f"ALERTS: {alerts}")
    
    def get_system_health_report(self):
        """Get comprehensive system health report"""
        if not self.metrics_history:
            return "No metrics collected yet"
        
        latest_metrics = self.metrics_history[-1]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'latest_metrics': latest_metrics,
            'total_metrics_collected': len(self.metrics_history),
            'active_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else [],  # Last 5 alerts
            'system_status': 'healthy' if not self.alerts or all('High' not in alert for alert in self.alerts[-5:]) else 'degraded'
        }
        
        return report

# Example usage
monitor = SystemMonitor()
current_metrics = monitor.collect_system_metrics()
print(f"Current metrics: {current_metrics}")

health_report = monitor.get_system_health_report()
print(f"Health report: {health_report}")
```

## Key Takeaways

The conversational humanoid robot capstone project demonstrates the complete integration of all components learned throughout the textbook:

- **Physical AI Foundation**: Embodied intelligence that interacts with the real world
- **ROS 2 Ecosystem**: Robust middleware for component orchestration
- **Simulation Integration**: Testing in both Gazebo and Isaac Sim environments
- **NVIDIA Isaac Stack**: Advanced perception and manipulation capabilities
- **Multimodal AI**: Combining vision, language, and other sensory inputs
- **Conversational Interface**: Natural human-robot interaction
- **System Integration**: Comprehensive architecture connecting all components

### Success Factors

1. **Modular Design**: Separate components for perception, cognition, and action
2. **Robust Communication**: Reliable ROS 2 messaging between components
3. **Adaptive Behavior**: Systems that respond to environmental changes
4. **Safety First**: Multiple safety layers and validation checks
5. **User-Centered**: Natural interaction that matches human expectations

## Future Enhancements

The foundational system described in this capstone can be extended with:

- **Advanced Learning**: Reinforcement learning for improved task execution
- **Social Intelligence**: Understanding and responding to social cues
- **Long-term Autonomy**: Extended operation with self-monitoring and maintenance
- **Multi-robot Systems**: Coordination with other robots
- **Cloud Integration**: Offloading computation-intensive tasks
- **Improved Safety**: Advanced safety mechanisms and emergency responses

## Conclusion

This capstone project synthesizes all the knowledge from the textbook into a functional conversational humanoid robot system. It demonstrates how Physical AI principles translate into real robotic capabilities that can understand natural language, navigate environments, manipulate objects, and interact naturally with humans.

The project showcases the integration of multiple complex systems into a cohesive whole, following best practices for system design, safety, and user experience. This serves as a foundation for further development of advanced humanoid capabilities.