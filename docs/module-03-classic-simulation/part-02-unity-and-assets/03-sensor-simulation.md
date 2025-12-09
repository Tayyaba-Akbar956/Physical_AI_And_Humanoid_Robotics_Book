---
sidebar_position: 5
title: Sensor Simulation
---

# Sensor Simulation

This chapter explores the simulation of robot sensors in both Gazebo and Unity environments, focusing on how to accurately model various sensor types to provide realistic robot perception in simulation.

## Learning Objectives

- Understand how different sensor types are simulated in robotics environments
- Implement camera, LIDAR, IMU, and other sensor models
- Evaluate the accuracy and limitations of sensor simulation
- Apply noise and error models to make simulations more realistic
- Integrate simulated sensors with ROS/ROS 2 for robot perception

## Introduction: The Senses of Robots

Sensors are the primary interface between robots and their environment. In simulation, these sensors must be carefully modeled to provide realistic data that enables effective algorithm development and testing. Sensor simulation bridges the gap between the virtual environment and the robot's perception system, providing the data needed for navigation, mapping, manipulation, and other robot capabilities.

Realistic sensor simulation must account for:
- Physical properties of the sensing modality
- Environmental factors affecting sensor performance
- Intrinsic sensor limitations and noise characteristics
- Integration with the physics simulation and environment rendering

### Sensor Categories in Robotics

**Active Sensors**: Emit energy and measure reflections
- LIDAR: Uses laser light for distance measurement
- Sonar: Uses sound waves for distance measurement
- Radar: Uses radio waves for detection

**Passive Sensors**: Capture ambient energy
- Cameras: Capture light in the visible spectrum
- Thermal cameras: Capture infrared radiation
- GPS: Receive signals from satellites

**Inertial Sensors**: Measure robot's own motion
- IMUs: Measure acceleration and rotation
- Odometry: Track wheel rotations or motion

**Proprioceptive Sensors**: Measure internal robot state
- Joint encoders: Measure joint angles
- Force/torque sensors: Measure forces at joints or end effectors

## Core Concepts

### Sensor Accuracy vs. Realism

Sensor simulation must balance accuracy (physical correctness) with realism (matching real sensor behavior):

**Accuracy** considerations:
- Physical modeling of sensor measurement process
- Proper ray casting, light transport, or signal propagation
- Accurate geometric relationships

**Realism** considerations:
- Noise and uncertainty similar to real sensors
- Environmental effects (fog, lighting, temperature)
- Latency and timing characteristics

### Simulation Fidelity Levels

Different levels of simulation fidelity are appropriate for different purposes:

**Low Fidelity** (Fast simulation):
- Simplified physics
- Reduced noise models
- Fast approximate algorithms
- Useful for rapid algorithm testing

**Medium Fidelity** (Balanced):
- Realistic physics modeling
- Basic noise and uncertainty
- Standard environmental effects
- Suitable for most development tasks

**High Fidelity** (Realistic simulation):
- Detailed physics modeling
- Comprehensive noise models
- Complex environmental effects
- Close approximation to real sensors

## Practical Implementation

### Camera Simulation in Gazebo

Camera sensors in Gazebo are implemented using the libgazebo_ros_camera plugin:

```xml
<!-- Adding a camera sensor to a robot in URDF/SDF -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <camera name="head">
      <horizontal_fov>1.089</horizontal_fov> <!-- 62.4 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <always_on>true</always_on>
    <update_rate>30.0</update_rate>
    <visualize>true</visualize>
    
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>camera</namespace>
        <remapping>image_raw:=image</remapping>
        <remapping>camera_info:=camera_info</remapping>
      </ros>
      <camera_name>camera</camera_name>
      <image_topic_name>image</image_topic_name>
      <camera_info_topic_name>camera_info</camera_info_topic_name>
      <frame_name>camera_optical_frame</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### LIDAR Simulation in Gazebo

LIDAR sensors simulate laser range finders:

```xml
<!-- Adding a 2D LIDAR to a robot -->
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>  <!-- -π radians -->
          <max_angle>3.14159</max_angle>   <!-- π radians -->
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
    
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>lidar</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Simulation in Gazebo

Inertial measurement units measure acceleration and angular velocity:

```xml
<!-- Adding an IMU to a robot -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <visualize>false</visualize>
    <update_rate>100</update_rate>
    
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.00174533</stddev> <!-- 0.1 deg/s -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.00174533</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.00174533</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>imu</namespace>
      </ros>
      <frame_name>imu_frame</frame_name>
      <body_name>imu_link</body_name>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### Camera Simulation in Unity

Camera sensors in Unity can simulate various visual sensors:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera cam;
    public string rosTopic = "/unity_camera/image_raw";
    public int publishRate = 30; // Hz
    
    [Header("Image Settings")]
    public int imageWidth = 640;
    public int imageHeight = 480;
    
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private ROSConnection ros;
    private int frameCount = 0;
    private int frameSkip;
    
    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(rosTopic);
        
        // Calculate frame skip based on desired rate
        frameSkip = Mathf.Max(1, Mathf.RoundToInt(60.0f / publishRate)); // Assuming 60 FPS
        
        SetupCamera();
    }
    
    void SetupCamera()
    {
        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        cam.targetTexture = renderTexture;
        
        // Create texture for reading
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }
    
    void Update()
    {
        if (frameCount % frameSkip == 0)
        {
            PublishImage();
        }
        frameCount++;
    }
    
    void PublishImage()
    {
        // Read pixels from render texture
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();
        RenderTexture.active = null;
        
        // Flip the image vertically to match ROS coordinate system
        Color[] pixels = texture2D.GetPixels();
        Color[] flippedPixels = new Color[pixels.Length];
        
        for (int y = 0; y < imageHeight; y++)
        {
            for (int x = 0; x < imageWidth; x++)
            {
                int originalIndex = y * imageWidth + x;
                int flippedIndex = (imageHeight - 1 - y) * imageWidth + x;
                flippedPixels[originalIndex] = pixels[flippedIndex];
            }
        }
        
        texture2D.SetPixels(flippedPixels);
        texture2D.Apply();
        
        // Encode texture as PNG bytes
        byte[] imageBytes = texture2D.EncodeToPNG();
        
        // Create ROS message
        ImageMsg msg = new ImageMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "camera_frame"
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(imageWidth * 3), // 3 bytes per pixel for RGB
            data = imageBytes
        };
        
        // Publish the image
        ros.Publish(rosTopic, msg);
    }
}
```

### LIDAR Simulation in Unity

Simulating LIDAR in Unity requires raycasting to determine distances:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityLIDARSensor : MonoBehaviour
{
    [Header("LIDAR Settings")]
    public string rosTopic = "/unity_lidar/scan";
    public int publishRate = 10; // Hz
    public float scanRange = 30.0f;
    public int horizontalResolution = 360;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public LayerMask detectionLayers = -1; // Detect all layers
    
    private ROSConnection ros;
    private float[] ranges;
    private float[] intensities;
    private int frameCount = 0;
    private int frameSkip;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<LaserScanMsg>(rosTopic);
        
        // Calculate frame skip based on desired rate (assuming ~60 FPS)
        frameSkip = Mathf.Max(1, Mathf.RoundToInt(60.0f / publishRate));
        
        // Initialize range arrays
        ranges = new float[horizontalResolution];
        intensities = new float[horizontalResolution];
    }
    
    void Update()
    {
        if (frameCount % frameSkip == 0)
        {
            PerformLIDARScan();
            PublishLIDARData();
        }
        frameCount++;
    }
    
    void PerformLIDARScan()
    {
        float angleStep = (maxAngle - minAngle) / horizontalResolution;
        
        for (int i = 0; i < horizontalResolution; i++)
        {
            float angle = minAngle + (i * angleStep);
            
            // Calculate direction in world space
            Vector3 direction = new Vector3(
                Mathf.Cos(angle),
                0f,  // For 2D LIDAR, keep Y = 0
                Mathf.Sin(angle)
            );
            
            // Rotate to match sensor orientation
            direction = transform.TransformDirection(direction);
            
            // Raycast to detect obstacles
            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, scanRange, detectionLayers))
            {
                ranges[i] = hit.distance;
                intensities[i] = 1.0f; // Simple intensity model
            }
            else
            {
                ranges[i] = scanRange; // No obstacle detected
                intensities[i] = 0.0f;
            }
        }
    }
    
    void PublishLIDARData()
    {
        LaserScanMsg msg = new LaserScanMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "lidar_frame"
            },
            angle_min = minAngle,
            angle_max = maxAngle,
            angle_increment = (maxAngle - minAngle) / horizontalResolution,
            time_increment = 0, // 2D scan - no time increment
            scan_time = 1.0f / publishRate,
            range_min = 0.1f,
            range_max = scanRange,
            ranges = ranges,
            intensities = intensities
        };
        
        ros.Publish(rosTopic, msg);
    }
    
    // Visualization in the Unity editor
    void OnDrawGizmosSelected()
    {
        if (ranges == null) return;
        
        float angleStep = (maxAngle - minAngle) / horizontalResolution;
        
        Gizmos.color = Color.red;
        for (int i = 0; i < horizontalResolution; i++)
        {
            float angle = minAngle + (i * angleStep);
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);
            
            if (ranges[i] < scanRange)
            {
                Gizmos.DrawRay(transform.position, direction * ranges[i]);
            }
            else
            {
                Gizmos.DrawRay(transform.position, direction * scanRange);
            }
        }
    }
}
```

### Noise Modeling

Real sensors have noise and uncertainty that should be modeled in simulation:

```csharp
using UnityEngine;

public class SensorNoiseModel
{
    // Add Gaussian noise to sensor reading
    public static float AddGaussianNoise(float value, float mean, float stddev)
    {
        // Box-Muller transform to generate Gaussian random numbers
        float u1 = Random.Range(0.0000001f, 1f); // Avoid log(0)
        float u2 = Random.Range(0f, 1f);
        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return value + mean + stddev * normal;
    }
    
    // Add bias to sensor reading
    public static float AddBias(float value, float bias)
    {
        return value + bias;
    }
    
    // Add drift to sensor reading
    public static float AddDrift(float value, float driftRate, float deltaTime, float referenceTime)
    {
        float drift = driftRate * (Time.time - referenceTime) * deltaTime;
        return value + drift;
    }
    
    // Apply noise model to a set of values
    public static float[] AddNoiseToScan(float[] scan, float noiseStddev)
    {
        float[] noisyScan = new float[scan.Length];
        for (int i = 0; i < scan.Length; i++)
        {
            // Only add noise to valid readings (not max range)
            if (scan[i] < 29.0f) // assuming max range is 30m
            {
                noisyScan[i] = AddGaussianNoise(scan[i], 0, noiseStddev);
                // Ensure no negative ranges
                noisyScan[i] = Mathf.Max(0.05f, noisyScan[i]);
            }
            else
            {
                noisyScan[i] = scan[i]; // Keep max range values
            }
        }
        return noisyScan;
    }
}
```

### Sensor Fusion Simulation

Simulating multiple sensors working together:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SensorFusionSystem : MonoBehaviour
{
    [Header("Sensors")]
    public UnityCameraSensor cameraSensor;
    public UnityLIDARSensor lidarSensor;
    public UnityIMUSensor imuSensor;
    
    [Header("Fusion Settings")]
    public float fusionRate = 10.0f; // Hz
    private float lastFusionTime = 0f;
    
    void Update()
    {
        if (Time.time - lastFusionTime >= 1.0f / fusionRate)
        {
            PerformSensorFusion();
            lastFusionTime = Time.time;
        }
    }
    
    void PerformSensorFusion()
    {
        // Example: Simple particle filter for localization
        
        // Get sensor readings (simplified)
        float[] lidarReadings = lidarSensor.GetRanges(); // This would need to be implemented
        Vector3 imuReading = imuSensor.GetLinearAcceleration(); // This would need to be implemented
        
        // Perform fusion algorithm
        // This is a very simplified example
        Vector3 estimatedPose = EstimatePose(lidarReadings, imuReading);
        
        // Publish fused result
        PublishFusedResult(estimatedPose);
    }
    
    Vector3 EstimatePose(float[] lidarReadings, Vector3 imuReading)
    {
        // Implement your fusion algorithm here
        // For example, a particle filter, Kalman filter, or other Bayes filter
        
        // This is a placeholder implementation
        return new Vector3(0, 0, 0);
    }
    
    void PublishFusedResult(Vector3 pose)
    {
        // Publish the fused result to ROS or other systems
        Debug.Log($"Fused pose estimate: {pose}");
    }
}
```

## Sensor Accuracy Considerations

### Environmental Factors

Several environmental factors affect sensor performance and should be modeled:

**Lighting Conditions**:
- Camera performance varies with illumination
- LIDAR can be affected by reflective surfaces
- Weather conditions affect various sensors

**Weather Simulation**:
- Rain, fog, and snow affect camera and LIDAR
- Wind can affect IMU readings
- Temperature drift in electronic components

**Surface Properties**:
- Reflectivity affects LIDAR and camera
- Material properties affect contact sensors
- Texture affects visual feature detection

### Sensor Limitations

Real sensors have limitations that must be modeled:

**Field of View**: 
- Cameras have limited angular coverage
- LIDAR has minimum and maximum detectable ranges
- Blind spots exist for all sensors

**Resolution**:
- Limited spatial resolution (pixels, angular resolution)
- Limited temporal resolution (update rate)
- Limited radiometric resolution (dynamic range)

**Latency**:
- Processing time creates sensor delays
- Communication delays in networked systems
- Integration effects in physical sensors

## Validation and Calibration

### Simulation Validation

Validate that simulated sensors behave like real sensors:

- Compare simulated and real sensor data for the same environment
- Validate noise characteristics match real sensors
- Check temporal and spatial properties

### Calibration Procedures

Calibrate both real and simulated sensors:

- Intrinsic calibration (internal parameters)
- Extrinsic calibration (position and orientation relative to robot)
- Temporal calibration (synchronization)

## Hands-on Exercise

1. **Camera Simulation**: Create a Unity scene with a camera sensor and implement the code to publish camera images to a ROS topic.

2. **LIDAR Comparison**: Compare the LIDAR simulation in Gazebo and Unity for the same environment, noting differences and similarities.

3. **Noise Modeling**: Add realistic noise models to a simulated sensor and compare the results with and without noise.

4. **Multi-Sensor Fusion**: Simulate two different sensors on the same robot and implement a simple fusion technique.

5. **Validation Exercise**: Research how to validate sensor simulation results by comparing against physical models or real sensors.

## Key Takeaways

- Sensor simulation must balance accuracy and computational performance
- Different sensors (camera, LIDAR, IMU) have different implementation approaches
- Noise and uncertainty modeling is crucial for realistic simulation
- Environmental factors significantly affect sensor performance
- Validation against real sensors is essential for effective simulation
- Sensor fusion requires careful modeling of each component sensor

## Further Reading

- "Probabilistic Robotics" by Thrun, Burgard, and Fox (for sensor models)
- Gazebo Sensor Tutorial
- Unity Robotics Sensors Documentation
- "Robotics, Vision and Control" by Corke (for sensor modeling)

## Next Steps

Continue to Module 4 to learn about NVIDIA Isaac, the AI-Robot Brain platform.