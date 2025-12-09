---
sidebar_position: 4
title: Isaac Sim Setup
---

# Isaac Sim Setup for Physical AI Development

This chapter provides comprehensive instructions for installing and configuring NVIDIA Isaac Sim, a high-fidelity simulation environment for Physical AI and robotics applications. Isaac Sim leverages NVIDIA's Omniverse platform for photorealistic simulation and high-performance physics.

## Learning Objectives

- Install and configure NVIDIA Isaac Sim for Physical AI applications
- Set up Omniverse for multi-user collaboration
- Create complex simulation environments with realistic physics and rendering
- Integrate Isaac Sim with ROS 2 for seamless development workflows
- Optimize Isaac Sim performance for Physical AI workloads
- Troubleshoot common Isaac Sim issues

## Introduction: Isaac Sim in Physical AI Development

NVIDIA Isaac Sim is a high-fidelity simulation environment built on the Omniverse platform. For Physical AI development, Isaac Sim offers:

- **Photorealistic Rendering**: Advanced graphics for computer vision training
- **High-Fidelity Physics**: Accurate simulation of real-world physics
- **Synthetic Data Generation**: Creation of labeled training data for AI models
- **Hardware Acceleration**: GPU-accelerated simulation and rendering
- **ROS 2 Integration**: Seamless connection with ROS 2 applications
- **Realistic Sensor Simulation**: Accurate camera, LiDAR, IMU, and other sensors

Isaac Sim is particularly valuable for Physical AI because it enables:
- Training AI models on synthetic data before deployment to real robots
- Testing navigation and manipulation tasks in complex environments
- Validating perception systems with realistic sensor data
- Developing sim-to-real transfer techniques

## Prerequisites for Isaac Sim

### Hardware Requirements

**Minimum Requirements**:
- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 32 GB
- GPU: NVIDIA RTX 3060 (12GB) or better
- Storage: 30+ GB free space
- OS: Ubuntu 20.04/22.04 or Windows 10/11

**Recommended Requirements**:
- CPU: Intel Core i9 or AMD Ryzen 9
- RAM: 64+ GB
- GPU: NVIDIA RTX 4080/4090 or RTX A5000/A6000
- Storage: 1 TB+ SSD
- Network: Gigabit Ethernet for multi-user environments

### Software Prerequisites

```bash
# Check GPU and driver support
nvidia-smi

# Verify OpenGL support
glxinfo | grep "OpenGL renderer"

# Install required packages
sudo apt update
sudo apt install -y wget curl unzip build-essential
```

## Installing Isaac Sim

### Method 1: Docker Installation (Recommended)

The easiest way to install Isaac Sim is using Docker:

```bash
# Install Docker if not already installed
sudo apt install -y docker.io
sudo usermod -aG docker $USER
# Log out and back in to apply changes

# Install NVIDIA Container Toolkit
curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -sL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Pull Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.0.0

# Create a script to run Isaac Sim
cat << 'EOF' > ~/run_isaac_sim.sh
#!/bin/bash
xhost +local:docker
docker run -it --gpus all \
  --network=host \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --env="NVIDIA_Prime_RENDER_OFFLOAD=1" \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --env="NVIDIA_VISIBLE_DEVICES=all" \
  --volume="/home/$USER/.Xauthority:/root/.Xauthority:rw" \
  --volume="/home/$USER/isaac_sim_data:/isaac-sim-data:rw" \
  --shm-size="1g" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name isaac-sim \
  nvcr.io/nvidia/isaac-sim:4.0.0
EOF

chmod +x ~/run_isaac_sim.sh
```

### Method 2: Native Installation (Advanced)

For native installation, download the Isaac Sim package:

```bash
# Download Isaac Sim (requires NVIDIA Developer Account)
# Visit https://developer.nvidia.com/isaac-sim
# Download the appropriate version

# Extract the package
tar -xzf isaac-sim-4.0.0.tar.gz
cd isaac-sim-4.0.0

# Run the installer
./install.sh

# Add to environment
echo 'source ~/isaac-sim-4.0.0/setup_bash.sh' >> ~/.bashrc
source ~/.bashrc
```

### Method 3: Using Omniverse Launcher (GUI Method)

1. Visit https://omniverse.nvidia.com/
2. Download and install the Omniverse Launcher
3. Sign in with your NVIDIA Developer account
4. Install Isaac Sim from the Applications tab

## Initial Isaac Sim Configuration

### Running Isaac Sim

Using Docker (recommended):

```bash
# Create data directory
mkdir -p ~/isaac_sim_data

# Run Isaac Sim
~/run_isaac_sim.sh
```

### Basic Configuration

Once Isaac Sim is running, configure the following:

1. **Workspace Setup**: Set up your project workspace at `/isaac-sim-data/projects`
2. **Extensions**: Enable relevant extensions for robotics simulation
3. **Physics**: Configure physics settings for your application
4. **Rendering**: Adjust rendering quality based on performance needs

### Environment Variables

Add these to your `~/.bashrc` for consistent configuration:

```bash
# Isaac Sim configuration
export ISAAC_SIM_PATH=~/isaac-sim-4.0.0
export OMNI_URL="omniverse://localhost/NewIsaacSim/"

# Physics settings
export PHYSICS_UPDATE_RATE=60  # Hz
export FIXED_SUB_STEPS=8

# Rendering settings (balance quality vs performance)
export ENABLE_NVPROFILER=0
export OVD_BATCH_SIZE=100000
export OVD_BATCH_COUNT=8
```

## Understanding Isaac Sim Architecture

### Core Components

Isaac Sim is built on NVIDIA Omniverse and consists of:

- **USD (Universal Scene Description)**: Scene representation format
- **PhysX**: NVIDIA's physics engine
- **Omniverse Kit**: Core runtime framework
- **RTX Renderer**: High-fidelity rendering engine
- **Isaac Extensions**: Robotics-specific capabilities

### USD in Isaac Sim

USD organizes scenes hierarchically:

```
World
├── Assets (robots, objects, environments)
│   ├── Robots
│   │   ├── URDF Files
│   │   ├── USD Files
│   │   └── Meshes
│   ├── Environments
│   │   ├── Rooms
│   │   ├── Outdoor Scenes
│   │   └── Objects
│   └── Props
│       ├── Furniture
│       ├── Obstacles
│       └── Tools
└── Simulation Settings
    ├── Physics
    ├── Rendering
    └── Time
```

## Creating Robot Models for Isaac Sim

### Robot Model Structure

Create a robot model in `~/isaac_sim_data/robots/physical_ai_robot/`:

```
physical_ai_robot/
├── urdf/
│   └── physical_ai_robot.urdf
├── meshes/
│   ├── base_link.STL
│   ├── wheel_left.STL
│   └── wheel_right.STL
├── materials/
│   └── robot_materials.mdl
└── config/
    ├── robot.yaml
    └── controllers.yaml
```

### URDF for Isaac Sim

Create a URDF file optimized for Isaac Sim (`~/isaac_sim_data/robots/physical_ai_robot/urdf/physical_ai_robot.urdf`):

```xml
<?xml version="1.0"?>
<robot name="physical_ai_robot">
  <!-- Base link with realistic properties -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://physical_ai_robot/meshes/base_link.STL"/>
      </geometry>
      <material name="robot_gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://physical_ai_robot/meshes/base_link.STL"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="wheel_left">
    <visual>
      <geometry>
        <mesh filename="package://physical_ai_robot/meshes/wheel_left.STL"/>
      </geometry>
      <material name="wheel_black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="wheel_right">
    <visual>
      <geometry>
        <mesh filename="package://physical_ai_robot/meshes/wheel_right.STL"/>
      </geometry>
      <material name="wheel_black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="base_to_wheel_left" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="-0.2 0.15 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="base_to_wheel_right" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right"/>
    <origin xyz="-0.2 -0.15 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Additional sensors -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.02"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="base_to_camera" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.25 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Extension for Isaac Sim -->
  <gazebo reference="base_link">
    <material>Orange</material>
  </gazebo>

  <gazebo reference="wheel_left">
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
    <fdir1>1 0 0</fdir1>
    <maxVel>1.0</maxVel>
    <minDepth>0.001</minDepth>
  </gazebo>

  <gazebo reference="wheel_right">
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>
</robot>
```

## Isaac Sim - ROS 2 Integration

### Installing Isaac ROS Bridge

```bash
# For Docker setup, the bridge is already included
# For native setup, install the ROS bridge extension:

cd ~/isaac-sim-4.0.0
./python.sh -m pip install numpy msgpack

# Install ROS bridge components
./python.sh -m pip install rclpy geometry_msgs sensor_msgs nav_msgs
```

### ROS Bridge Configuration

Create ROS bridge configuration at `~/isaac_sim_data/config/ros_bridge_config.yaml`:

```yaml
# Isaac Sim ROS Bridge Configuration
bridge_config:
  # Robot description publisher
  robot_description:
    type: publisher
    topic: robot_description
    message_type: std_msgs/String
    path: /World/physical_ai_robot/URDF

  # Joint state publisher
  joint_states:
    type: publisher
    topic: joint_states
    message_type: sensor_msgs/JointState
    path: /World/physical_ai_robot/JointState

  # Camera image publisher
  camera_image:
    type: publisher
    topic: /camera/image_raw
    message_type: sensor_msgs/Image
    path: /World/physical_ai_robot/Camera
    encoding: rgb8

  # LiDAR scan publisher
  lidar_scan:
    type: publisher
    topic: /scan
    message_type: sensor_msgs/LaserScan
    path: /World/physical_ai_robot/LiDAR

  # Velocity command subscriber
  cmd_vel:
    type: subscriber
    topic: /cmd_vel
    message_type: geometry_msgs/Twist
    path: /World/physical_ai_robot/DiffDriveController
```

### Launching Isaac Sim with ROS Bridge

Create `~/isaac_sim_data/launch/isaac_sim_with_ros.py`:

```python
import carb
import omni
import omni.ext
import omni.kit.commands
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils import viewports
from omgi.isaac.core.utils import nucleus
import asyncio
import threading
import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
import time

class IsaacSimROSNode(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros_node')
        
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.get_logger().info('Isaac Sim ROS Bridge initialized')

    def cmd_vel_callback(self, msg):
        # This would send commands to Isaac Sim when properly connected
        self.get_logger().info(f'Received cmd_vel: {msg.linear.x}, {msg.angular.z}')

def run_ros_node():
    rclpy.init()
    node = IsaacSimROSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def setup_isaac_sim_world():
    """Setup Isaac Sim world with robot and sensors"""
    # Initialize world
    world = World(stage_units_in_meters=1.0)
    
    # Add robot
    get_assets_root_path()
    add_reference_to_stage(
        usd_path="path/to/physical_ai_robot.usd",
        prim_path="/World/physical_ai_robot",
        position=[0, 0, 0.5],
        orientation=[0, 0, 0, 1]
    )
    
    # Configure physics
    world.scene.add_default_ground_plane()
    
    # Set up camera
    # Set up LiDAR
    # Set up other sensors
    
    return world

# Main execution
if __name__ == "__main__":
    # Start ROS node in a separate thread
    ros_thread = threading.Thread(target=run_ros_node)
    ros_thread.start()
    
    # Setup Isaac Sim world
    world = setup_isaac_sim_world()
    
    # Run simulation
    world.reset()
    for i in range(1000):
        world.step(render=True)
    
    # Cleanup
    ros_thread.join()
```

## Creating Simulation Environments

### Environment Structure

Create environments in `~/isaac_sim_data/environments/`:

```
environments/
├── indoor/
│   ├── factory_floor.usd
│   ├── warehouse.usd
│   └── office.usd
├── outdoor/
│   ├── urban_street.usd
│   ├── rural_path.usd
│   └── construction_site.usd
└── specialized/
    ├── grasping_lab.usd
    ├── navigation_test.usd
    └── perception_challenge.usd
```

### Creating an Indoor Environment

Create `~/isaac_sim_data/environments/indoor/physical_ai_lab.usd`:

```usda
#usda 1.0
(
    customLayerData = {
        string creatorName = "Isaac Sim Physical AI Lab"
        string creatorVersion = "4.0.0"
    }
    defaultPrim = "World"
    subLayers = [
        @./_sublayers/ground_plane.usda@
    ]
)

def Xform "World"
{
    def Xform "Environment"
    {
        def Xform "Furniture"
        {
            def Xform "Table"
            {
                # Table properties
                add references = @./assets/table.usd@</Table>
                over "Table"
                {
                    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
                }
            }
        }
        
        def Xform "Obstacles"
        {
            def Xform "Box1"
            {
                add references = @./assets/box.usd@</Box>
            }
            def Xform "Cylinder1"
            {
                add references = @./assets/cylinder.usd@</Cylinder>
            }
        }
    }
    
    def Xform "Robot"
    {
        add references = @./../robots/physical_ai_robot.usd@</Robot>
        over "Robot"
        {
            prepend apiSchemas = [
                "DifferentialController",
                "IsaacSensor"
            ]
        }
    }
    
    # Physics properties
    def PhysicsScene "physicsScene"
    {
        float3 gravity = (0, 0, -9.81)
        float physicsTimeStep = 0.008333  # 120 Hz
    }
}
```

## Configuring Sensors in Isaac Sim

### Camera Sensor Configuration

Add a realistic camera to your robot in Isaac Sim:

```python
# Python script to configure camera in Isaac Sim
import omni
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.sensor import Camera
from omni.isaac.core import World
import numpy as np

def setup_camera(robot_path, camera_path="/World/physical_ai_robot/Camera"):
    """Setup RGB camera on the robot"""
    # Create the camera prim
    camera = Camera(
        prim_path=camera_path,
        frequency=30,  # Hz
        resolution=(640, 480)
    )
    
    # Position the camera
    camera.set_position(np.array([0.25, 0, 0.2]))
    camera.set_rotation(np.array([0, 0, 0, 1]))
    
    # Configure camera properties
    camera.post_process_enabled = True
    camera.focal_length = 24.0  # mm
    camera.focus_distance = 10.0  # m
    camera.f_stop = 0.0  # Infinity focus
    camera.horizontal_aperture = 20.955  # mm
    camera.vertical_aperture = 15.2908  # mm
    
    return camera

def capture_camera_data(camera):
    """Capture and process camera data"""
    rgb_data = camera.get_rgb()
    depth_data = camera.get_depth()
    seg_data = camera.get_semantic_segmentation()
    
    return rgb_data, depth_data, seg_data
```

### LiDAR Configuration

Add LiDAR sensors for 3D perception:

```python
from omni.isaac.range_sensor import _range_sensor
import omni
import carb

def setup_lidar(robot_path, lidar_path="/World/physical_ai_robot/LiDAR"):
    """Setup LiDAR sensor on the robot"""
    # Get the range sensor interface
    lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
    
    # Create LiDAR sensor
    lidar_config = {
        "rotation_frequency": 10,  # Hz
        "points_per_second": 540000,
        "horizontal_samples": 16,
        "horizontal_fov": 360,
        "vertical_fov": 30,
        "vertical_angle_offsets": [0] * 16,
        "min_range": 0.1,
        "max_range": 25.0,
        "return_mode": _range_sensor.ReturnMode.Distance,
        "sensor_x": 0.2,  # X offset from robot
        "sensor_y": 0.0,
        "sensor_z": 0.3,
        "sensor_yaw": 0.0,
        "sensor_pitch": 0.0,
        "sensor_roll": 0.0
    }
    
    # Create the LiDAR prim
    lidar_interface.create_range_sensor(
        lidar_path,
        10,  # update frequency
        carb.Float3(50, 50, 50),  # translation
        carb.Float3(0, 0, 0),     # rotation
        "RPLIDAR_A1",             # sensor name
        360,                      # horizontal samples
        16,                       # vertical samples
        5,                        # horizontal_fov
        30,                       # vertical_fov
        0.1,                      # min_range
        25.0                      # max_range
    )
    
    return lidar_path
```

## Isaac Sim Performance Optimization

### Graphics Settings

Adjust graphics settings based on your hardware capability:

```python
# In Isaac Sim UI or through scripting
carb.settings.get_settings().set("/app/window/dpi_scale", 1.0)  # UI scale
carb.settings.get_settings().set("/rtx/post/dlss/enable", True)  # Enable DLSS if supported
carb.settings.get_settings().set("/rtx/quality/raytracing", 0)   # Reduce raytracing
carb.settings.get_settings().set("/rtx/quality/reflections", 0)  # Simplify reflections
carb.settings.get_settings().set("/renderer/quality", 0)         # Lower quality preset
```

### Physics Optimization

For performance-critical applications:

```python
# In Physics Scene settings
carb.settings.get_settings().set("/physics/fix_degenerate_joints", True)
carb.settings.get_settings().set("/physics/d6prismatic_spring", False)
carb.settings.get_settings().set("/physics/bounce_threshold", 2.0)  # m/s
carb.settings.get_settings().set("/physics/default_solver_position_iteration_count", 4)
carb.settings.get_settings().set("/physics/default_solver_velocity_iteration_count", 1)
```

### Headless Operation

For automated testing or CI/CD pipelines:

```bash
# Run Isaac Sim in headless mode
docker run --gpus all --rm -it \
  -v {LOCAL_PATH}/isaac_sim_data:/isaac-sim-data:rw \
  --shm-size="1g" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "PRIVACY_LEVEL=1" \
  -e "OMNIVERSE_HEADLESS=1" \
  nvcr.io/nvidia/isaac-sim:4.0.0
```

## Physical AI-Specific Configuration

### Photorealistic Environments for CV Training

For training computer vision models, configure Isaac Sim for synthetic data generation:

```python
import omni.replicator.core as rep

def setup_replicator_for_cv():
    """Setup Omniverse Replicator for synthetic data generation"""
    
    # Define training objects
    with rep.new_layer():
        # Randomize object positions
        robot = rep.get.prims(prim_types=['Xform'], name='physical_ai_robot')
        
        # Randomize lighting
        lights = rep.get.light()
        with lights.randomize.light_brightness(
            temperature_mean=6500,
            temperature_spread=500,
            intensity_mean=300,
            intensity_spread=50
        ):
            lights.light_color = rep.randomizer.color(colors=[
                (1.0, 0.9, 0.8),  # Warm white
                (0.9, 0.95, 1.0), # Cool white
            ])
        
        # Randomize materials
        objects = rep.get.prims(prim_types=['Geometry', 'Xform'])
        with objects.randomize.random_material(material=rep.utils.random_material()):
            pass
    
    # Generate dataset
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=f"/isaac-sim-data/cv_dataset_{int(time.time())}")
    writer.attach([rep.observations.camera_annotations_view])
```

### Physics-Based Learning Environments

Create environments specifically for reinforcement learning:

```python
def setup_rl_environment():
    """Setup environment for reinforcement learning"""
    world = World(stage_units_in_meters=1.0)
    
    # Add robot
    robot = world.scene.add(
        Robot(
            prim_path="/World/physical_ai_robot",
            name="physical_ai_robot",
            usd_path="path/to/robot.usd"
        )
    )
    
    # Configure physics for RL
    world.physics_sim_view.set_subspace_mass_matrix(False)
    world.set_physics_dt(1/60.0, substeps=4)  # 60Hz with 4 substeps
    
    # Add RL tasks
    # Navigation task
    # Manipulation task
    # Balancing task
    
    return world
```

## Troubleshooting Common Isaac Sim Issues

### Issue 1: GPU Memory Problems

**Problem**: Isaac Sim crashes with GPU memory errors

**Solutions**:
```bash
# Reduce rendering quality
export RTX_GLOBAL_TEXTURE_STREAMING_MEMORY_BUDGET_IN_MB=4096

# Reduce physics complexity
export PHYSICS_MESH_COMPLEXITY=2

# Use smaller textures
# In Isaac Sim UI: Window > Renderer > Quality Settings > Texture Resolution
```

### Issue 2: ROS Bridge Connection Failures

**Problem**: ROS topics don't connect properly

**Solutions**:
```bash
# Check ROS environment
printenv | grep ROS

# Verify network configuration
ifconfig
netstat -tuln

# Test basic ROS functionality
ros2 topic list
ros2 run demo_nodes_cpp talker
```

### Issue 3: Slow Simulation Performance

**Problem**: Simulation runs slower than real-time

**Solutions**:
```bash
# Reduce physics update rate
export PHYSICS_UPDATE_RATE=60

# Simplify collision meshes
# Use fewer objects in the scene
# Reduce sensor resolution

# Enable multi-threading
import omni
carb.settings.get_settings().set("/app/player/play_options", {"multi_threading": True})
```

### Issue 4: Docker Permission Errors

**Problem**: Isaac Sim fails to start in Docker with permissions

**Solutions**:
```bash
# Ensure user is in docker group
groups $USER

# Check X11 forwarding
xhost +local:docker

# Ensure NVIDIA runtime is configured
docker info | grep nvidia
```

## Isaac Sim Best Practices for Physical AI

### 1. Scene Organization

```bash
# Organize scenes by type
~/isaac_sim_data/scenes/
├── training_scenes/
│   ├── navigation_scenes/
│   ├── manipulation_scenes/
│   └── perception_scenes/
├── validation_scenes/
└── test_scenes/
```

### 2. Asset Management

- Use relative paths in USD files
- Keep asset files organized in a consistent directory structure
- Use version control for scene files
- Document scene variations and parameters

### 3. Reproducible Experiments

```python
import hashlib
import json

def create_experiment_config(seed, scene_path, robot_config):
    """Create reproducible experiment configuration"""
    config = {
        'seed': seed,
        'scene_path': scene_path,
        'robot_config': robot_config,
        'isaac_sim_version': '4.0.0',
        'creation_timestamp': time.time()
    }
    
    # Create hash for reproducibility
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config['experiment_id'] = config_hash
    
    return config
```

### 4. Synthetic Data Pipeline

```python
def generate_synthetic_dataset(scene_list, output_dir, num_samples_per_scene):
    """Generate synthetic dataset for Physical AI training"""
    for scene_path in scene_list:
        # Load scene
        load_scene(scene_path)
        
        # Add random objects
        add_random_objects()
        
        # Randomize lighting
        randomize_lighting()
        
        # Randomize materials
        randomize_materials()
        
        # Capture data
        for i in range(num_samples_per_scene):
            reset_scene()
            capture_frame_with_annotations(output_dir)
```

## Testing Isaac Sim Setup

### Basic Functionality Test

```bash
# Test Isaac Sim startup
~/run_isaac_sim.sh

# In Isaac Sim, verify:
# - Scene loads correctly
# - Physics simulation works
# - Rendering is smooth
# - Camera and sensors function
```

### ROS Integration Test

```python
# Test ROS communication with Isaac Sim
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

def test_ros_connection():
    rclpy.init()
    node = rclpy.create_node('isaac_sim_tester')
    
    # Publisher for robot commands
    cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
    
    # Subscriber for camera data
    def image_callback(msg):
        node.get_logger().info(f'Received image: {msg.width}x{msg.height}')
    
    img_sub = node.create_subscription(Image, '/camera/image_raw', image_callback, 10)
    
    # Send test command
    cmd = Twist()
    cmd.linear.x = 0.5
    cmd.angular.z = 0.2
    
    # Publish command and wait for response
    for i in range(10):
        cmd_pub.publish(cmd)
        rclpy.spin_once(node, timeout_sec=0.1)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    test_ros_connection()
```

## Advanced Isaac Sim Features for Physical AI

### Domain Randomization

```python
def setup_domain_randomization():
    """Setup domain randomization for sim-to-real transfer"""
    
    # Randomize physical properties
    with rep.randomizer.physics_material():
        rep.randomizer.dynamic_friction((0.1, 1.0))
        rep.randomizer.static_friction((0.1, 1.0))
        rep.randomizer.restitution((0.0, 0.5))
    
    # Randomize visual properties
    with rep.randomizer.material():
        rep.randomizer.roughness((0.0, 1.0))
        rep.randomizer.metallic((0.0, 1.0))
        rep.randomizer.diffuse_reflection((0.0, 1.0))
    
    # Randomize lighting
    with rep.randomizer.light():
        rep.randomizer.light_intensity((100, 1000))
        rep.randomizer.light_color([(0.8, 0.9, 1.0), (1.0, 0.9, 0.8)])
```

### Multi-Robot Simulation

```python
def setup_multi_robot_scene(robot_count=3):
    """Setup scene with multiple robots"""
    world = World(stage_units_in_meters=1.0)
    
    for i in range(robot_count):
        robot_path = f"/World/Robot_{i}"
        robot = world.scene.add(
            Robot(
                prim_path=robot_path,
                name=f"robot_{i}",
                usd_path="path/to/robot.usd",
                position=[i*2, 0, 0.5]  # Space robots apart
            )
        )
    
    return world
```

## Key Takeaways

- Isaac Sim provides high-fidelity simulation for advanced Physical AI development
- Docker installation is recommended for ease of setup and consistency
- Proper sensor configuration is critical for synthetic data generation
- Physics settings must be tuned for realistic behavior
- ROS integration enables seamless workflows with existing robotics code
- Performance optimization is necessary for complex scenes

## Further Reading

- "Isaac Sim Documentation": Official NVIDIA documentation
- "Omniverse Replicator Guide": Synthetic data generation techniques
- "Simulation-Based Learning for Robotics": Advanced simulation techniques
- "Domain Randomization for Robotics": Techniques for sim-to-real transfer

## Next Steps

Continue to Appendix B, Section 5: Troubleshooting to learn how to solve common issues in Physical AI development environments.