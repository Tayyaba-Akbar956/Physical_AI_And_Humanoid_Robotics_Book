---
sidebar_position: 4
title: GPT Integration
---

# GPT Integration

This chapter explores the integration of large language models (LLMs), particularly those similar to GPT, into robotics applications. These models provide advanced natural language understanding and generation capabilities that can enhance conversational robots, task planning, and human-robot interaction. The chapter covers both the technical aspects of integration and the practical considerations for deployment in physical systems.

## Learning Objectives

- Understand the architecture and capabilities of transformer-based language models
- Implement GPT-like models for robotics applications
- Design safe and effective integration patterns for LLMs in robots
- Evaluate the performance and limitations of LLM integration
- Address safety and ethical considerations in LLM deployment

## Introduction: LLMs in Physical AI

Large Language Models (LLMs) like GPT have revolutionized natural language processing, demonstrating remarkable capabilities in understanding and generating human-like text. For physical AI systems, these models offer opportunities to enhance:

- **Natural Language Understanding**: Better comprehension of complex, nuanced human instructions
- **Conversational Abilities**: More natural, context-aware interactions
- **Task Planning**: High-level reasoning and planning based on natural language
- **Knowledge Integration**: Access to vast amounts of world knowledge
- **Adaptability**: Learning and adapting through interaction with minimal explicit programming

However, integrating LLMs into robotics presents unique challenges:

- **Real-time Constraints**: LLMs may not meet the timing requirements for physical interaction
- **Reliability**: LLMs can produce unpredictable outputs that may be unsafe for physical systems
- **Grounding**: Connecting abstract language understanding to concrete physical actions
- **Latency**: Network calls to cloud-based models can introduce unacceptable delays
- **Safety**: Ensuring LLM outputs don't lead to unsafe robot behaviors

### LLM Capabilities in Robotics

**Language Understanding**: Processing complex, multi-sentence instructions
**Knowledge Access**: Answering questions about the world, science, or procedures
**Reasoning**: Logical inference, problem-solving, and planning
**Generation**: Creating appropriate responses and explanations
**Adaptation**: Learning from interaction and improving over time

### Integration Approaches

**Cloud-based Services**: Using APIs to access pre-trained models
**Edge Deployment**: Running models on robot's local compute hardware
**Hybrid Approach**: Combining local and remote processing
**Specialized Models**: Fine-tuning general LLMs for specific robotic tasks

## Core Concepts

### Transformer Architecture

**Self-Attention Mechanism**: Allows the model to focus on different parts of input
**Positional Encoding**: Incorporates information about token position in sequence
**Feed-Forward Networks**: Process each position independently
**Layer Normalization and Residual Connections**: Improve training stability

### Context Window and Grounding

**Context Window**: The maximum length of text the model can process at once
**Prompt Engineering**: Crafting inputs to guide model behavior effectively
**Grounding**: Connecting language to physical reality and robot state
**Chain-of-Thought Reasoning**: Multi-step reasoning for complex tasks

### Safety and Alignment

**Safety Filtering**: Preventing generation of harmful content
**Reliability**: Ensuring consistent, predictable behavior
**Bias Mitigation**: Reducing harmful biases in responses
**Validation**: Verifying that instructions are executable and safe

## Practical Implementation

### LLM Integration Framework

```python
import openai
import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

@dataclass
class LLMConfig:
    api_key: str = ""
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7
    timeout: int = 30
    base_url: Optional[str] = None  # For local models

class LLMInterface:
    def __init__(self, config: LLMConfig):
        self.config = config
        openai.api_key = config.api_key
        if config.base_url:
            openai.base_url = config.base_url
        
        # Conversation history (for models that benefit from context)
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10  # Limit history size
    
    def set_system_context(self, system_prompt: str):
        """Set the system context/prompt"""
        # Remove existing system message if present
        self.conversation_history = [
            msg for msg in self.conversation_history 
            if msg.get("role") != "system"
        ]
        # Add new system message at the beginning
        self.conversation_history.insert(0, {
            "role": "system", 
            "content": system_prompt
        })
    
    def query_model(self, prompt: str) -> Optional[str]:
        """Query the LLM with a single prompt"""
        try:
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message['content'].strip()
        
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None
    
    def query_with_context(self, user_input: str) -> Optional[str]:
        """Query the LLM maintaining conversation context"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user", 
            "content": user_input
        })
        
        try:
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=self.conversation_history,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            response_text = response.choices[0].message['content'].strip()
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": response_text
            })
            
            # Maintain history size limit
            if len(self.conversation_history) > self.max_history_length:
                # Keep system message if present, remove oldest non-system messages
                system_idx = None
                for i, msg in enumerate(self.conversation_history):
                    if msg.get("role") == "system":
                        system_idx = i
                        break
                
                if system_idx is not None:
                    # Preserve system message and newer messages
                    preserved_messages = [self.conversation_history[system_idx]]
                    preserved_messages.extend(
                        self.conversation_history[-(self.max_history_length-1):]
                    )
                    self.conversation_history = preserved_messages
                else:
                    # No system message, just limit to max length
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            return response_text
        
        except Exception as e:
            print(f"Error querying LLM with context: {e}")
            # Remove the user message that caused the error
            if self.conversation_history and self.conversation_history[-1].get("role") == "user":
                self.conversation_history.pop()
            return None

# Example usage
if __name__ == "__main__":
    config = LLMConfig(
        api_key="YOUR_API_KEY",  # Replace with actual API key
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    llm = LLMInterface(config)
    
    # Set system context for a robot assistant
    llm.set_system_context(
        "You are a helpful robot assistant. You can help with navigation, "
        "answering questions, and performing simple tasks. Always be polite "
        "and acknowledge when you cannot perform physical actions directly."
    )
    
    # Simple query
    response = llm.query_model("What is 2+2?")
    print(f"Response: {response}")
    
    # Context-based query
    response = llm.query_with_context("Hello, how can you help me?")
    print(f"Contextual response: {response}")
```

### Task Planning with LLMs

```python
import json
from typing import Dict, List, Any

class LLMTaskPlanner:
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        
        # Robot capabilities and environment knowledge
        self.robot_capabilities = [
            "move to location",
            "grasp object",
            "manipulate object",
            "navigate environment",
            "answer questions",
            "perform inspection",
            "transport object"
        ]
        
        self.environment_knowledge = {
            "locations": ["kitchen", "living room", "bedroom", "office", "entrance"],
            "objects": ["cup", "book", "bottle", "box", "phone", "remote"],
            "actions": ["pick up", "put down", "move", "grasp", "transport"]
        }
    
    def parse_task_request(self, user_request: str, robot_state: Dict) -> Optional[Dict]:
        """Parse a natural language task request into structured format"""
        system_prompt = f"""
        You are a task parsing assistant for a robot. Parse the user's request into structured format.
        
        Robot capabilities: {', '.join(self.robot_capabilities)}
        Environment: {json.dumps(self.environment_knowledge)}
        Current robot state: {json.dumps(robot_state)}
        
        Return a JSON object with the following format:
        {{
            "intent": "command type (navigate, manipulate, answer, etc.)",
            "primary_object": "main object of interest",
            "destination": "location if applicable",
            "action": "specific action to take",
            "confidence": "confidence in parsing (0-1)",
            "reasoning": "brief explanation of your interpretation"
        }}
        
        Keep responses concise and in valid JSON format. Do not include any text outside the JSON object.
        """
        
        user_prompt = f"Parse this request: {user_request}"
        
        # Set system context and query
        self.llm.set_system_context(system_prompt)
        response = self.llm.query_model(user_prompt)
        
        if response:
            try:
                # Extract JSON from response (in case model includes extra text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    parsed_task = json.loads(json_str)
                    return parsed_task
            except json.JSONDecodeError:
                print(f"Could not parse JSON from LLM response: {response}")
                return None
        
        return None
    
    def generate_task_plan(self, parsed_task: Dict, robot_state: Dict) -> Optional[List[Dict]]:
        """Generate a step-by-step plan for the parsed task"""
        system_prompt = f"""
        You are a task planning assistant for a robot. Given the parsed task and robot state,
        generate a step-by-step plan to accomplish the task.
        
        Parsed task: {json.dumps(parsed_task)}
        Robot state: {json.dumps(robot_state)}
        Robot capabilities: {', '.join(self.robot_capabilities)}
        
        Return a JSON array with steps in this format:
        [
            {{
                "step": 1,
                "action": "action_type",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "description": "human-readable description",
                "estimated_time": "estimated time in seconds"
            }}
        ]
        
        Ensure the plan is executable by the robot and safe.
        """
        
        self.llm.set_system_context(system_prompt)
        response = self.llm.query_model("Generate a task plan.")
        
        if response:
            try:
                # Extract JSON from response
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    task_plan = json.loads(json_str)
                    return task_plan
            except json.JSONDecodeError:
                print(f"Could not parse JSON from LLM response: {response}")
                return None
        
        return None
    
    def execute_task_with_llm(self, user_request: str, robot_state: Dict) -> Dict:
        """Execute a task by querying LLM for understanding and planning"""
        # Step 1: Parse the request
        parsed_task = self.parse_task_request(user_request, robot_state)
        if not parsed_task:
            return {
                "status": "error",
                "message": "Could not understand the task request",
                "plan": []
            }
        
        # Step 2: Generate plan
        task_plan = self.generate_task_plan(parsed_task, robot_state)
        if not task_plan:
            return {
                "status": "error", 
                "message": "Could not generate a plan for the task",
                "plan": [],
                "parsed_task": parsed_task
            }
        
        return {
            "status": "success",
            "message": f"Task understood and plan generated ({len(task_plan)} steps)",
            "plan": task_plan,
            "parsed_task": parsed_task
        }

# Example usage
if __name__ == "__main__":
    # This assumes you have an LLMInterface configured
    # For example with a mock or actual configuration:
    
    class MockLLMInterface:
        def __init__(self):
            pass
        
        def set_system_context(self, context):
            pass
        
        def query_model(self, prompt):
            # Mock response for parsing
            if "Parse this request" in prompt:
                return '{"intent": "manipulate", "primary_object": "cup", "destination": "kitchen", "action": "transport", "confidence": 0.8, "reasoning": "User wants to move cup to kitchen"}'
            elif "Generate a task plan" in prompt:
                return """[
                    {"step": 1, "action": "navigate", "parameters": {"destination": "location_of_cup"}, "description": "Move to the cup's location", "estimated_time": 15},
                    {"step": 2, "action": "grasp", "parameters": {"object": "cup"}, "description": "Pick up the cup", "estimated_time": 10},
                    {"step": 3, "action": "navigate", "parameters": {"destination": "kitchen"}, "description": "Move to kitchen", "estimated_time": 20},
                    {"step": 4, "action": "place", "parameters": {"object": "cup", "location": "counter"}, "description": "Place cup on counter", "estimated_time": 10}
                ]"""
            return ""
    
    # Create a planner with the mock interface
    mock_llm = MockLLMInterface()
    planner = LLMTaskPlanner(mock_llm)
    
    # Example robot state
    robot_state = {
        "location": "living room",
        "battery_level": 85,
        "carrying_object": None,
        "last_known_object_locations": {
            "cup": "coffee table in living room"
        }
    }
    
    # Execute a task
    result = planner.execute_task_with_llm("Please take the cup from the coffee table to the kitchen", robot_state)
    print(f"Task execution result: {json.dumps(result, indent=2)}")
```

### Safety and Validation Layer

```python
from enum import Enum
import re

class SafetyLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"

class LLMSafetyValidator:
    def __init__(self):
        # Dangerous commands to filter
        self.dangerous_keywords = [
            "harm", "injure", "damage", "break", "destroy", "unsafe",
            "dangerous", "hurt", "attack", "fight", "weapon", "fire",
            "explosive", "electric shock", "fall", "trap", "choke"
        ]
        
        # Restricted actions for safety
        self.restricted_actions = [
            "self_harm", "harm_others", "damage_property", 
            "violate_privacy", "bypass_safety", "ignore_emergency"
        ]
    
    def validate_response(self, response: str, robot_capabilities: List[str]) -> SafetyLevel:
        """Validate LLM response for safety issues"""
        response_lower = response.lower()
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in response_lower:
                return SafetyLevel.UNSAFE
        
        # Check for restricted actions or commands
        if any(action in response_lower for action in self.restricted_actions):
            return SafetyLevel.UNSAFE
        
        # Additional checks could include:
        # - Commands that bypass safety systems
        # - Requests that violate ethical guidelines
        # - Instructions that could cause physical harm
        
        return SafetyLevel.SAFE
    
    def validate_task_plan(self, task_plan: List[Dict]) -> SafetyLevel:
        """Validate a task plan for safety issues"""
        for step in task_plan:
            action = step.get('action', '').lower()
            params = step.get('parameters', {})
            
            # Check if action is in robot's safe capabilities
            if action not in ['navigate', 'grasp', 'place', 'transport', 'inspect', 'answer']:
                return SafetyLevel.UNSAFE
            
            # Check for potentially unsafe parameter values
            if 'destination' in params:
                dest = params['destination'].lower()
                if any(danger_zone in dest for danger_zone in ['danger', 'unsafe', 'hazard']):
                    return SafetyLevel.CAUTION
            
            # Additional safety checks would go here
            # For example, validating that destinations are safe
        
        return SafetyLevel.SAFE
    
    def apply_safety_filters(self, response: str) -> str:
        """Apply safety filters to LLM response"""
        # Basic content filtering
        filtered_response = response
        
        # Replace potentially harmful content with safe alternatives
        dangerous_patterns = [
            (r"(harm|injure|damage|destroy)\s+the\s+(\w+)", r"assist with the \2"),
            (r"(attack|fight|aggressive)\s+mode", r"assistance mode"),
        ]
        
        for pattern, replacement in dangerous_patterns:
            filtered_response = re.sub(pattern, replacement, filtered_response, flags=re.IGNORECASE)
        
        return filtered_response

# Example usage
if __name__ == "__main__":
    validator = LLMSafetyValidator()
    
    # Test response validation
    safe_response = "I can help you navigate to the kitchen safely."
    unsafe_response = "I will harm the object."
    
    print(f"Safe response validation: {validator.validate_response(safe_response, [])}")
    print(f"Unsafe response validation: {validator.validate_response(unsafe_response, [])}")
    
    # Test task plan validation
    safe_plan = [
        {"action": "navigate", "parameters": {"destination": "kitchen"}},
        {"action": "grasp", "parameters": {"object": "cup"}}
    ]
    
    unsafe_plan = [
        {"action": "harm_others", "parameters": {"target": "person"}},
        {"action": "damage", "parameters": {"object": "property"}}
    ]
    
    print(f"Safe plan validation: {validator.validate_task_plan(safe_plan)}")
    print(f"Unsafe plan validation: {validator.validate_task_plan(unsafe_plan)}")
```

### Grounding Language in Physical Reality

```python
from typing import Optional
import math

class PhysicalGroundingSystem:
    def __init__(self):
        # Store mapping between linguistic references and physical entities
        self.entity_map = {}
        self.spatial_relations = {}
        self.current_environment = {}
    
    def ground_spatial_reference(self, reference: str, robot_location: List[float], 
                                 environment_map: Dict) -> Optional[List[float]]:
        """Ground a spatial reference to a physical location"""
        reference_lower = reference.lower()
        
        # Handle relative spatial references
        if reference_lower in ['here', 'this location', 'current position']:
            return robot_location
        
        elif reference_lower in ['kitchen', 'living room', 'bedroom', 'office', 'entrance']:
            # Look up in environment map
            if reference_lower in environment_map:
                return environment_map[reference_lower].get('location')
        
        elif 'left' in reference_lower or 'right' in reference_lower:
            # Parse relative directions (simplified)
            direction = 'left' if 'left' in reference_lower else 'right'
            distance = self._extract_distance(reference_lower) or 1.0  # Default 1m if not specified
            
            # Calculate new position (simplified 2D)
            # In practice, this would use the robot's orientation
            new_pos = robot_location.copy()
            if direction == 'left':
                new_pos[0] -= distance  # X decreases to the left
            else:
                new_pos[0] += distance  # X increases to the right
            return new_pos
        
        elif 'behind' in reference_lower or 'in front of' in reference_lower:
            # Handle front/back relative to robot orientation
            # This would require knowledge of robot's current heading
            pass
        
        return None  # Could not ground the reference
    
    def ground_object_reference(self, reference: str, environment_objects: List[Dict]) -> Optional[Dict]:
        """Ground an object reference to a physical object in the environment"""
        reference_lower = reference.lower()
        
        # Find the most likely matching object
        best_match = None
        best_score = 0
        
        for obj in environment_objects:
            obj_name = obj.get('name', '').lower()
            obj_type = obj.get('type', '').lower()
            obj_desc = obj.get('description', '').lower()
            
            # Calculate match score based on various factors
            score = 0
            
            if reference_lower == obj_name:
                score += 10
            elif reference_lower == obj_type:
                score += 8
            elif reference_lower in obj_name:
                score += 6
            elif reference_lower in obj_type:
                score += 5
            elif reference_lower in obj_desc:
                score += 3
            elif obj_name in reference_lower or obj_type in reference_lower:
                score += 2
            
            # Also consider color, size, and distinctive attributes
            obj_attributes = obj.get('attributes', {})
            for attr_key, attr_val in obj_attributes.items():
                if isinstance(attr_val, str) and attr_val.lower() in reference_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = obj
        
        # Only return if we have a reasonably good match
        if best_score >= 5:
            return best_match
        else:
            return None
    
    def _extract_distance(self, text: str) -> Optional[float]:
        """Extract distance value from text (simplified)"""
        import re
        
        # Look for patterns like "2 meters", "1.5m", "3 feet", etc.
        patterns = [
            r'(\d+(?:\.\d+)?)\s*meters?',  # e.g., "2 meters", "1.5m"
            r'(\d+(?:\.\d+)?)\s*m',        # e.g., "1.5m"
            r'(\d+(?:\.\d+)?)\s*feet?',    # e.g., "3 feet"
            r'(\d+(?:\.\d+)?)\s*ft',       # e.g., "3ft"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None  # No distance found
    
    def resolve_coreferences(self, conversation_history: List[Dict], 
                           environment_objects: List[Dict]) -> List[Dict]:
        """Resolve pronouns and references in conversation history"""
        resolved_history = []
        
        # Keep track of mentioned objects and locations
        last_mentioned_objects = []
        last_mentioned_locations = []
        
        for turn in conversation_history:
            text = turn.get('text', '')
            resolved_text = text
            
            # Resolve "it", "that", "the object", etc.
            if 'it' in text.lower() or 'that' in text.lower():
                if last_mentioned_objects:
                    # Replace "it" or "that" with the last mentioned object
                    last_obj = last_mentioned_objects[-1]
                    resolved_text = resolved_text.lower().replace('it', last_obj.get('name', 'object'))
                    resolved_text = resolved_text.lower().replace('that', last_obj.get('name', 'object'))
            
            # Create resolved turn
            resolved_turn = turn.copy()
            resolved_turn['resolved_text'] = resolved_text
            resolved_history.append(resolved_turn)
            
            # Update history of mentioned objects/locations
            # This is a simplified approach - in practice, you'd have more sophisticated coreference resolution
            found_objects = []
            for obj in environment_objects:
                obj_name = obj.get('name', '').lower()
                if obj_name in text.lower():
                    found_objects.append(obj)
            
            if found_objects:
                last_mentioned_objects.extend(found_objects)
        
        return resolved_history

# Example usage
if __name__ == "__main__":
    grounding_system = PhysicalGroundingSystem()
    
    # Example environment
    env_objects = [
        {'name': 'red cup', 'type': 'cup', 'location': [1.0, 2.0, 0], 'attributes': {'color': 'red', 'material': 'ceramic'}},
        {'name': 'book', 'type': 'book', 'location': [1.5, 2.5, 0], 'attributes': {'color': 'blue', 'material': 'paper'}},
        {'name': 'plant', 'type': 'plant', 'location': [3.0, 1.0, 0], 'attributes': {'type': 'indoor', 'size': 'medium'}}
    ]
    
    env_map = {
        'kitchen': {'location': [5.0, 0.0, 0]},
        'living room': {'location': [0.0, 0.0, 0]},
        'bedroom': {'location': [8.0, 4.0, 0]}
    }
    
    robot_pos = [0.5, 0.5, 0]
    
    # Test spatial grounding
    kitchen_loc = grounding_system.ground_spatial_reference("kitchen", robot_pos, env_map)
    print(f"Grounded 'kitchen' to location: {kitchen_loc}")
    
    # Test object grounding
    target_obj = grounding_system.ground_object_reference("red cup", env_objects)
    print(f"Grounded 'red cup' to object: {target_obj}")
    
    # Test "it" reference
    conversation = [
        {'text': "Pick up the red cup", 'role': 'user'},
        {'text': "Put it on the table", 'role': 'user'}
    ]
    
    resolved = grounding_system.resolve_coreferences(conversation, env_objects)
    print(f"Resolved conversation: {resolved}")
```

### Integration with Robot Systems

```python
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class LLMRobotController:
    def __init__(self, llm_interface: LLMInterface, task_planner: LLMTaskPlanner, 
                 validator: LLMSafetyValidator, grounding_system: PhysicalGroundingSystem):
        self.llm = llm_interface
        self.planner = task_planner
        self.validator = validator
        self.grounding = grounding_system
        
        # Robot state and API
        self.robot_state = {
            'location': [0.0, 0.0, 0.0],
            'battery_level': 100,
            'carrying_object': None,
            'current_task': None,
            'available': True
        }
        
        self.robot_api = None  # Will be set to actual robot control API
    
    def set_robot_api(self, robot_api):
        """Set the robot API for physical control"""
        self.robot_api = robot_api
    
    async def process_command(self, user_command: str) -> Dict:
        """Process a user command end-to-end through the LLM system"""
        start_time = time.time()
        
        # 1. Update robot state
        self._update_robot_state()
        
        # 2. Parse the command using LLM
        parsed_task = self.planner.parse_task_request(
            user_command, 
            self.robot_state
        )
        
        if not parsed_task:
            return {
                "status": "error",
                "message": "Could not understand the command",
                "execution_time": time.time() - start_time,
                "success": False
            }
        
        # 3. Validate for safety
        safety_check = self.validator.validate_response(
            json.dumps(parsed_task), 
            self.robot_state.get('capabilities', [])
        )
        
        if safety_check == SafetyLevel.UNSAFE:
            return {
                "status": "unsafe",
                "message": "The requested action is unsafe",
                "execution_time": time.time() - start_time,
                "success": False
            }
        
        # 4. Generate detailed task plan
        task_plan = self.planner.generate_task_plan(parsed_task, self.robot_state)
        
        if not task_plan:
            return {
                "status": "error",
                "message": "Could not generate a task plan",
                "execution_time": time.time() - start_time,
                "parsed_task": parsed_task,
                "success": False
            }
        
        # 5. Validate the task plan for safety
        plan_safety = self.validator.validate_task_plan(task_plan)
        
        if plan_safety == SafetyLevel.UNSAFE:
            return {
                "status": "unsafe",
                "message": "The generated task plan contains unsafe actions",
                "execution_time": time.time() - start_time,
                "task_plan": task_plan,
                "success": False
            }
        
        # 6. Ground language in physical reality
        grounded_plan = await self._ground_plan_in_reality(task_plan)
        
        # 7. Execute the plan (in simulation for this example)
        execution_result = await self._execute_plan(grounded_plan)
        
        return {
            "status": "success" if execution_result['success'] else "execution_failed",
            "message": execution_result['message'],
            "execution_time": time.time() - start_time,
            "parsed_task": parsed_task,
            "task_plan": grounded_plan,
            "success": execution_result['success']
        }
    
    async def _ground_plan_in_reality(self, task_plan: List[Dict]) -> List[Dict]:
        """Ground abstract plan steps in physical reality"""
        grounded_plan = []
        
        for step in task_plan:
            grounded_step = step.copy()
            
            # Ground spatial references
            if 'destination' in step.get('parameters', {}):
                dest_ref = step['parameters']['destination']
                grounded_pos = self.grounding.ground_spatial_reference(
                    dest_ref, 
                    self.robot_state['location'], 
                    self.current_environment_map()
                )
                
                if grounded_pos:
                    grounded_step['parameters']['grounded_destination'] = grounded_pos
                else:
                    grounded_step['parameters']['grounded_destination'] = step['parameters']['destination']
            
            # Ground object references
            if 'object' in step.get('parameters', {}):
                obj_ref = step['parameters']['object']
                grounded_obj = self.grounding.ground_object_reference(
                    obj_ref, 
                    self.current_environment_objects()
                )
                
                if grounded_obj:
                    grounded_step['parameters']['grounded_object'] = grounded_obj
                else:
                    grounded_step['parameters']['grounded_object'] = {'name': obj_ref}
            
            grounded_plan.append(grounded_step)
        
        return grounded_plan
    
    async def _execute_plan(self, grounded_plan: List[Dict]) -> Dict:
        """Execute the grounded task plan (simulated)"""
        results = []
        
        for step in grounded_plan:
            step_result = await self._execute_single_step(step)
            results.append(step_result)
            
            # Update robot state after each step
            self._update_robot_state()
        
        success = all(r['success'] for r in results)
        message = f"Executed {len(results)} steps" + (", all successful" if success else ", with some failures")
        
        return {
            "success": success,
            "message": message,
            "step_results": results
        }
    
    async def _execute_single_step(self, step: Dict) -> Dict:
        """Execute a single step of the task plan"""
        action = step.get('action', '').lower()
        params = step.get('parameters', {})
        
        print(f"Executing: {action} with params {params}")
        
        # Simulate action execution with delays
        if action == 'navigate':
            await asyncio.sleep(0.5)  # Simulate navigation time
            # Update robot location if destination is specified
            dest = params.get('grounded_destination') or params.get('destination')
            if dest:
                self.robot_state['location'] = dest if isinstance(dest, list) else [0, 0, 0]
        
        elif action == 'grasp':
            await asyncio.sleep(0.3)  # Simulate grasping time
            obj = params.get('grounded_object') or {'name': 'unknown'}
            self.robot_state['carrying_object'] = obj.get('name', 'object')
        
        elif action == 'place':
            await asyncio.sleep(0.3)  # Simulate placing time
            self.robot_state['carrying_object'] = None
        
        # Mark step as successful
        return {
            "step": step.get('step', 0),
            "action": action,
            "success": True,
            "details": f"Completed {action} action"
        }
    
    def _update_robot_state(self):
        """Update the robot state from the physical robot"""
        # In a real system, this would query the actual robot for current state
        # For simulation, we'll just add some variation
        self.robot_state['battery_level'] = max(0, self.robot_state['battery_level'] - 0.1)
        self.robot_state['timestamp'] = datetime.now().isoformat()
    
    def current_environment_objects(self) -> List[Dict]:
        """Get current environment objects (simulated)"""
        # In a real system, this would come from perception systems
        return [
            {'name': 'red cup', 'type': 'cup', 'location': [1.0, 2.0, 0], 'attributes': {'color': 'red'}},
            {'name': 'blue book', 'type': 'book', 'location': [1.5, 2.5, 0], 'attributes': {'color': 'blue'}}
        ]
    
    def current_environment_map(self) -> Dict:
        """Get current environment map (simulated)"""
        # In a real system, this would come from mapping systems
        return {
            'kitchen': {'location': [5.0, 0.0, 0]},
            'living room': {'location': [0.0, 0.0, 0]},
            'bedroom': {'location': [8.0, 4.0, 0]}
        }

# Example usage
if __name__ == "__main__":
    import time
    
    # Create an instance of the system (using mocks for LLM components)
    class MockLLMInterface:
        def set_system_context(self, context): pass
        def query_model(self, prompt): return '{"intent": "navigate", "primary_object": "cup", "destination": "kitchen", "action": "transport", "confidence": 0.8, "reasoning": "User wants to move cup to kitchen"}'
    
    class MockTaskPlanner:
        def parse_task_request(self, request, state): return {"intent": "transport", "object": "cup", "destination": "kitchen"}
        def generate_task_plan(self, parsed_task, robot_state): return [{"step": 1, "action": "navigate", "parameters": {"destination": "kitchen"}}]
    
    # Create the controller with mocks
    mock_llm = MockLLMInterface()
    mock_planner = MockTaskPlanner()
    validator = LLMSafetyValidator()
    grounding_system = PhysicalGroundingSystem()
    
    controller = LLMRobotController(mock_llm, mock_planner, validator, grounding_system)
    
    # Process a command
    result = asyncio.run(controller.process_command("Take the red cup to the kitchen"))
    print(f"Command result: {json.dumps(result, indent=2)}")
```

## Advanced Integration Techniques

### Local Model Deployment

```python
# Example for running local models (like Llama) with transformers
try:
    import transformers
    import torch
    
    class LocalLLMInterface:
        def __init__(self, model_name="microsoft/DialoGPT-medium"):
            """
            Initialize a local LLM interface using transformers
            """
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token if not defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def query_model(self, prompt: str, max_length: int = 100) -> str:
            """
            Query the local model
            """
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, 
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (after the prompt)
            response = response[len(prompt):].strip()
            return response
    
    print("Local LLM interface available")
except ImportError:
    print("Transformers library not available, skipping local model interface")

# Example for using ONNX Runtime for optimized inference
try:
    import onnxruntime as ort
    
    class OptimizedLLMInterface:
        def __init__(self, model_path: str):
            """
            Initialize LLM interface with ONNX runtime for optimized inference
            """
            self.session = ort.InferenceSession(model_path)
            # Additional initialization would depend on the specific model
        
        def query_model(self, prompt: str) -> str:
            """
            Query the optimized local model
            """
            # Implementation would depend on the specific ONNX model
            # This is a placeholder
            return f"Response to: {prompt}"
    
    print("ONNX-based LLM interface available")
except ImportError:
    print("ONNX Runtime not available, skipping optimized interface")
```

### Context Management and Memory

```python
import pickle
from datetime import datetime, timedelta

class ContextualLLMManager:
    def __init__(self, llm_interface: LLMInterface, max_context_tokens: int = 4000):
        self.llm = llm_interface
        self.max_context_tokens = max_context_tokens
        
        # Conversation memory
        self.conversations = {}  # Per-user conversation history
        self.global_context = {}  # Cross-user context
        self.context_summaries = {}  # Summarized long-term memory
        
        # Memory management parameters
        self.compression_threshold = 0.8  # When to compress memory
        self.summarization_interval = 10  # Summarize every N turns
    
    def add_to_conversation(self, user_id: str, role: str, content: str):
        """Add a message to the conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        
        self.conversations[user_id].append(message)
        
        # Check if we should summarize the conversation
        if len(self.conversations[user_id]) % self.summarization_interval == 0:
            self._summarize_conversation(user_id)
        
        # Apply memory pressure management
        self._manage_memory_pressure(user_id)
    
    def _summarize_conversation(self, user_id: str):
        """Create a summary of the conversation for long-term memory"""
        if user_id not in self.conversations or not self.conversations[user_id]:
            return
        
        # Get recent conversation to summarize
        recent_messages = self.conversations[user_id][-5:]  # Last 5 messages
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_messages
        ])
        
        # Query LLM to create a summary
        summary_prompt = f"""
        Summarize the following conversation in 1-2 sentences, focusing on the main topics and outcomes:
        
        {conversation_text}
        
        Summary:
        """
        
        summary = self.llm.query_model(summary_prompt)
        
        if summary:
            # Store summary with timestamp
            summary_entry = {
                "summary": summary,
                "timestamp": datetime.now(),
                "conversation_length": len(self.conversations[user_id])
            }
            
            if user_id not in self.context_summaries:
                self.context_summaries[user_id] = []
            
            self.context_summaries[user_id].append(summary_entry)
    
    def _manage_memory_pressure(self, user_id: str):
        """Manage memory pressure by compressing or forgetting old context"""
        # Estimate token count (rough approximation: 1 token ~ 4 characters)
        conversation = self.conversations[user_id]
        estimated_tokens = sum(len(msg['content']) // 4 for msg in conversation)
        
        if estimated_tokens > self.max_context_tokens * self.compression_threshold:
            # Remove oldest messages while preserving system context
            while (sum(len(msg['content']) // 4 for msg in conversation) > 
                   self.max_context_tokens * 0.6 and len(conversation) > 3):
                # Don't remove system messages
                if conversation[0].get('role') != 'system':
                    conversation.pop(0)
                else:
                    conversation.pop(1)  # Remove the second message instead
    
    def get_context_for_query(self, user_id: str) -> List[Dict[str, str]]:
        """Get appropriate context for an LLM query"""
        context = []
        
        # Add long-term summary if available
        if user_id in self.context_summaries and self.context_summaries[user_id]:
            latest_summary = self.context_summaries[user_id][-1]
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {latest_summary['summary']}"
            })
        
        # Add recent conversation history
        if user_id in self.conversations:
            # Include recent messages up to token limit
            recent_messages = self.conversations[user_id][-10:]  # Last 10 messages
            context.extend(recent_messages)
        
        return context
    
    def query_with_memory(self, user_id: str, user_input: str) -> str:
        """Query LLM with relevant context from memory"""
        # Add user input to conversation
        self.add_to_conversation(user_id, "user", user_input)
        
        # Get relevant context
        context = self.get_context_for_query(user_id)
        
        # Query the LLM with context
        # In practice, you'd need to format this according to your LLM's requirements
        # This is a simplified approach
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        
        response = self.llm.query_model(full_prompt + f"\nassistant:")
        
        if response:
            self.add_to_conversation(user_id, "assistant", response)
        
        return response
```

## Troubleshooting Common Issues

### Performance Problems

**High Latency**: Use local models or edge deployment, implement caching for common requests
**Memory Usage**: Implement context window management and memory compression
**Throughput**: Use model optimization, batching, or multiple instances

### Reliability Issues

**Inconsistent Outputs**: Use lower temperature, implement output validation
**Context Loss**: Implement proper context management and conversation memory
**Safety Violations**: Deploy multiple safety layers and validation checks

### Grounding Problems

**Abstract to Physical**: Implement comprehensive grounding systems with perception feedback
**World Model Drift**: Regularly update environment models and verify actions
**Reference Resolution**: Deploy robust coreference and spatial reasoning systems

## Best Practices

### Integration Design

- Implement layered safety with multiple validation steps
- Use appropriate model sizes for compute constraints
- Design for graceful degradation when LLM is unavailable
- Maintain clear separation between LLM outputs and robot actions

### Safety and Ethics

- Always validate LLM outputs before robot action
- Implement explicit approval for uncertain commands
- Maintain logs for audit and improvement
- Consider bias and fairness in LLM outputs

### Performance Optimization

- Cache responses for common queries
- Use model quantization for edge deployment
- Implement asynchronous processing where possible
- Monitor and optimize response times

## Hands-on Exercise

1. **LLM Integration**: Implement the basic LLM interface and integrate it with a robot simulation.

2. **Task Planning**: Create an LLM-based task planning system that converts natural language to executable robot actions.

3. **Safety Validation**: Implement a comprehensive safety validation system for LLM outputs.

4. **Physical Grounding**: Develop systems to ground language in physical reality using environment perception.

5. **Context Management**: Create a context management system that maintains conversation history and long-term memory.

## Key Takeaways

- LLMs can significantly enhance robot natural language understanding and generation
- Safety and validation are critical when integrating LLMs with physical systems
- Context management is essential for coherent, multi-turn interactions
- Grounding abstract language in physical reality requires specialized systems
- Performance and reliability challenges must be addressed for practical deployment
- Local model deployment can improve privacy and reduce latency

## Further Reading

- "Language Models and Robotics" - Recent research papers
- "Safety in AI Systems" - Safety engineering for AI systems
- "Embodied AI" - Research at the intersection of AI and robotics
- "Transformer Architectures" - Technical details of modern LLMs

## Next Steps

Continue to Chapter 5: Multimodal Integration to explore how LLMs can be enhanced with visual and other sensory modalities for richer robot interaction.