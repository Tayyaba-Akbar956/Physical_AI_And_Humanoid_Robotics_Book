---
sidebar_position: 1
title: Software Installation
---

# Software Installation for Physical AI & Humanoid Robotics

This chapter provides comprehensive instructions for installing all necessary software to follow the Physical AI & Humanoid Robotics textbook. The installation process is one of the most complex aspects of robotics development, so we provide detailed instructions for multiple platforms and scenarios.

## Learning Objectives

- Install and configure ROS 2 development environment
- Set up simulation tools (Gazebo, Isaac Sim)
- Configure NVIDIA Isaac SDK components
- Establish a functional development workflow
- Troubleshoot common installation issues

## Introduction: Robotics Software Stack

Physical AI development requires a complex software stack that includes:

- **ROS 2**: Robot Operating System for communication and middleware
- **Simulation tools**: Gazebo, Isaac Sim, or Unity for simulation environments
- **AI frameworks**: PyTorch, TensorFlow, or other ML libraries
- **Development tools**: IDEs, version control, debugging tools
- **Hardware interfaces**: Drivers for specific robot platforms

The following instructions are designed to work on Ubuntu 22.04 LTS, which is the recommended environment for this textbook.

## Prerequisites

### System Requirements

**Minimum Configuration**:
- 64-bit processor with 4+ cores
- 8 GB RAM (16+ GB recommended)
- 50 GB free disk space
- 1024x768 display resolution

**Recommended Configuration**:
- 64-bit processor with 6+ cores (Intel i7 or AMD Ryzen 7)
- 32+ GB RAM
- 100+ GB SSD storage
- Dedicated GPU with 4+ GB VRAM (NVIDIA preferred)
- 1920x1080 or higher display resolution

### Operating System Support

**Primary Support (Recommended)**:
- Ubuntu 22.04 LTS (Jammy Jellyfish)
- ROS 2 Humble Hawksbill (LTS version)

**Alternative Support**:
- Windows 10/11 with WSL2 (Ubuntu 22.04)
- macOS 12+ (with Docker or virtual machine)

## Installing Ubuntu 22.04 LTS (If Needed)

### Option 1: Native Installation

1. **Download Ubuntu 22.04 LTS**:
   - Visit https://ubuntu.com/download/desktop
   - Download the 22.04 LTS version
   - Verify checksum before installation

2. **Create Bootable USB**:
   ```bash
   # On Ubuntu/MacOS
   sudo dd if=ubuntu-22.04-desktop-amd64.iso of=/dev/sdX bs=4M status=progress

   # On Windows (using Rufus or similar tool)
   # Follow tool-specific instructions
   ```

3. **Boot from USB and Install**:
   - Restart computer and enter boot menu (usually F12 or Esc during startup)
   - Select USB device to boot from
   - Follow Ubuntu installation wizard
   - Recommended: select "Install third-party software"

### Option 2: Virtual Machine

1. **Install Virtualization Software**:
   - VirtualBox (free): https://www.virtualbox.org/
   - VMware Workstation Player (free for personal use)
   - Hyper-V (Windows Pro/Enterprise)

2. **Configure VM Settings**:
   - Memory: 8+ GB (16+ GB preferred)
   - CPU: 2+ cores (4+ cores preferred)
   - Storage: 50+ GB (100+ GB recommended)
   - Enable 3D acceleration
   - Enable nested virtualization if available

### Option 3: WSL2 (Windows Only)

For Windows users, WSL2 provides a Linux development environment:

```bash
# Open PowerShell as Administrator and run:
wsl --install

# Or manually:
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart your computer
# Download and install Linux kernel update package from Microsoft
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

## Installing ROS 2 Humble Hawksbill

### Step 1: Set Up Sources

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install locale settings
sudo apt install -y locales
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe

# Add ROS 2 GPG key and repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros-key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Step 2: Install ROS 2 Packages

```bash
# Update package lists after adding ROS 2 repository
sudo apt update

# Install ROS 2 Humble desktop packages
sudo apt install -y ros-humble-desktop

# Install additional ROS 2 tools
sudo apt install -y ros-humble-ros-base
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
sudo apt install -y ros-humble-xacro ros-humble-joint-state-publisher-gui
sudo apt install -y ros-humble-robot-state-publisher ros-humble-controller-manager
sudo apt install -y ros-humble-ros2-control ros-humble-ros2-controllers
```

### Step 3: Install Python Dependencies

```bash
# Install pip if not already installed
sudo apt install -y python3-pip

# Install Python dependencies for robotics development
pip3 install -U argcomplete
pip3 install -U numpy matplotlib
pip3 install -U opencv-contrib-python
pip3 install -U transforms3d
pip3 install -U pyquaternion
pip3 install -U open3d
pip3 install -U pybullet
```

### Step 4: Initialize rosdep

```bash
# Initialize rosdep
sudo rosdep init
rosdep update

# Set up ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Installing Simulation Tools

### Gazebo Garden (Recommended)

Gazebo Garden is the latest version of the Gazebo simulator:

```bash
# Add Gazebo repository
curl -sSL http://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install gazebo-garden
```

### Alternative: Gazebo Classic

For compatibility with existing tutorials:

```bash
sudo apt install ros-humble-gazebo-classic
# Or install from source if needed
```

### NVIDIA Isaac Sim (Optional but Recommended)

For advanced Physical AI development:

1. **Install NVIDIA Container Toolkit**:
   ```bash
   # Add NVIDIA package repository
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   
   curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=containerd
   sudo systemctl restart containerd
   ```

2. **Install Docker**:
   ```bash
   sudo apt install -y docker.io
   sudo usermod -aG docker $USER
   # Log out and back in to apply changes
   ```

3. **Pull Isaac Sim Docker Image**:
   ```bash
   docker pull nvcr.io/nvidia/isaac-sim:4.0.0
   ```

## Installing Development Tools

### Python Development Environment

```bash
# Install Python-related development tools
sudo apt install -y python3-dev python3-venv python3-pil python3-pil.imagetk
sudo apt install -y python3-numpy python3-matplotlib python3-scipy python3-pandas
```

### Integrated Development Environment (IDE)

**Option 1: VS Code with ROS Extension**:
```bash
# Download and install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install -y code

# Install ROS extension in VS Code
# Press Ctrl+Shift+P, type "Extensions: Install Extensions", search "ROS"
```

**Option 2: PyCharm**:
```bash
# Install PyCharm (Community or Professional)
sudo snap install pycharm-community --classic
```

### Version Control

```bash
# Install Git and Git GUI tools
sudo apt install -y git git-gui gitk
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## NVIDIA GPU Setup (If Applicable)

For GPU-accelerated robotics applications:

```bash
# Check if NVIDIA GPU is present
lspci | grep -i nvidia

# Install NVIDIA drivers (use Software & Updates GUI or command line)
sudo apt install -y nvidia-driver-535  # Or latest stable version

# Verify installation
nvidia-smi

# Install CUDA (optional, for advanced GPU programming)
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run
```

Add CUDA to environment:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Testing the Installation

### Basic ROS 2 Test

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Test basic functionality
ros2 --version
ros2 topic list
```

### Python ROS 2 Test

```bash
# Create a temporary directory for testing
mkdir -p ~/ros2_test && cd ~/ros2_test

# Test Python ROS 2 modules
python3 -c "import rclpy; print('rclpy import successful')"
python3 -c "from geometry_msgs.msg import Twist; print('geometry_msgs import successful')"
python3 -c "from sensor_msgs.msg import LaserScan; print('sensor_msgs import successful')"
```

### Gazebo Test

```bash
# Test Gazebo (if installed)
gz sim --version
# Or for Gazebo Classic:
gazebo --version

# Launch Gazebo in background to test GUI
gz sim -s ServerGUIPlugin &
sleep 5
pkill gz
```

## Workspace Setup

### Create ROS 2 Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source ROS 2 before building
source /opt/ros/humble/setup.bash

# Build workspace (should build with no packages initially)
colcon build --symlink-install

# Source the workspace
source install/setup.bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Common Installation Issues and Solutions

### Issue 1: Permission Denied Errors

**Problem**: Commands fail with permission denied errors
**Solution**: Use `sudo` appropriately, or check user group membership:
```bash
groups $USER
# Ensure user is in needed groups (docker, dialout, etc.)
```

### Issue 2: Package Installation Failures

**Problem**: APT fails to install packages
**Solution**: Update package lists and fix broken dependencies:
```bash
sudo apt update
sudo apt upgrade
sudo apt --fix-broken install
```

### Issue 3: Python Import Errors

**Problem**: Python cannot import ROS 2 modules
**Solution**: Ensure environment is properly sourced and Python paths are correct:
```bash
echo $PYTHONPATH
source /opt/ros/humble/setup.bash
python3 -c "import sys; print(sys.path)"
```

### Issue 4: Missing Dependencies

**Problem**: Missing package dependencies during builds
**Solution**: Install dependencies using rosdep:
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### Issue 5: Gazebo Not Launching

**Problem**: Gazebo fails to start with graphics errors
**Solution**: Check graphics drivers and environment:
```bash
glxinfo | grep -i "direct rendering"
export LIBGL_ALWAYS_SOFTWARE=1  # Use software rendering if needed
```

## Alternative Installation Methods

### Using Docker (For Isolated Environment)

```bash
# Pull ROS 2 Humble development image
docker pull osrf/ros:humble-desktop-full

# Run with GUI support (Linux)
xhost +local:docker
docker run -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --device=/dev/dri:/dev/dri:rw \
  --name=ros2_dev \
  osrf/ros:humble-desktop-full
```

### Using Conda Environment (For Python Management)

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create conda environment
conda create -n ros2_env python=3.10
conda activate ros2_env

# Install Python packages in conda environment
conda install numpy matplotlib scipy pandas
```

## Verification Checklist

After completing the installation, verify the following:

- [ ] ROS 2 Humble is installed and accessible
- [ ] Python modules import correctly
- [ ] Gazebo simulator launches successfully
- [ ] Workspace builds without errors
- [ ] Basic ROS 2 tutorials run correctly
- [ ] GPU acceleration is available (if applicable)
- [ ] Development tools are accessible

## Post-Installation Configuration

### Environment Variables

Add the following to `~/.bashrc` for persistent configuration:

```bash
# ROS 2 Configuration
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Gazebo Configuration
export GZ_SIM_SYSTEM_PLUGIN_PATH="~/ros2_ws/install/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH}"

# Python Configuration
export PYTHONPATH="~/ros2_ws/install/lib/python3.10/site-packages:$PYTHONPATH"

# Colcon Configuration
export COLCON_CURRENT_PREFIX="~/ros2_ws/install"
```

Then reload the bash configuration:

```bash
source ~/.bashrc
```

## Updating the Installation

### Regular Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update ROS 2 packages
sudo apt update && sudo apt install ros-humble-desktop

# Update Python packages
pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U
```

### Backup and Restore

```bash
# Create a list of installed packages
dpkg --get-selections | grep ros > ros_packages_backup.txt

# Create a backup of your workspace
tar -czf ros2_workspace_backup.tar.gz ~/ros2_ws
```

## Troubleshooting Resources

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- ROS Discourse: https://discourse.ros.org/
- Gazebo Tutorials: http://gazebosim.org/tutorials
- NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/

## Key Takeaways

- The software installation for Physical AI development is complex but essential
- Ubuntu 22.04 with ROS 2 Humble is the recommended configuration
- Proper environment setup is crucial for successful development
- Regular updates help maintain system security and compatibility
- Having backup plans (Docker, cloud alternatives) is helpful for troubleshooting
- Take time to verify each component works before proceeding

## Further Reading

- "ROS 2 Documentation - Installation" - Complete installation guide
- "Gazebo Installation Guide" - Official simulator documentation
- "Setting up Development Environment with VS Code" - IDE configuration
- NVIDIA Isaac Sim Installation Guide

## Next Steps

Continue to Appendix B, Section 2: ROS 2 Setup to configure ROS 2 for robotics development.