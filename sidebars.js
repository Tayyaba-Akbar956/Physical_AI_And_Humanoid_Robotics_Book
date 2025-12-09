// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro'],
      collapsed: false, // Keep "Introduction" visible/open initially
    },
    {
      type: 'category',
      label: 'Module 1: Foundations',
      items: [
        {
          type: 'category',
          label: 'Part 1: Core Concepts',
          items: [
            'module-01-foundations/part-01-concepts/intro-physical-ai',
            'module-01-foundations/part-01-concepts/embodied-intelligence',
            'module-01-foundations/part-01-concepts/digital-vs-physical'
          ],
        },
        {
          type: 'category',
          label: 'Part 2: Landscape',
          items: [
            'module-01-foundations/part-02-landscape/humanoid-landscape',
            'module-01-foundations/part-02-landscape/sensor-systems'
          ],
        },
      ],
      collapsed: true, // All categories collapsed by default
    },
    {
      type: 'category',
      label: 'Module 2: ROS 2 - The Robotic Nervous System',
      items: [
        {
          type: 'category',
          label: 'Part 1: Communication',
          items: [
            'module-02-ros2-middleware/part-01-communication/ros2-overview',
            'module-02-ros2-middleware/part-01-communication/nodes-topics-services',
            'module-02-ros2-middleware/part-01-communication/python-rclpy'
          ],
        },
        {
          type: 'category',
          label: 'Part 2: Robot Description',
          items: [
            'module-02-ros2-middleware/part-02-robot-description/urdf-format',
            'module-02-ros2-middleware/part-02-robot-description/launch-files'
          ],
        },
      ],
      collapsed: true, // All categories collapsed by default
    },
    {
      type: 'category',
      label: 'Module 3: Simulation - Digital Twins',
      items: [
        {
          type: 'category',
          label: 'Part 1: Gazebo',
          items: [
            'module-03-classic-simulation/part-01-gazebo/gazebo-intro',
            'module-03-classic-simulation/part-01-gazebo/physics-simulation'
          ],
        },
        {
          type: 'category',
          label: 'Part 2: Unity and Assets',
          items: [
            'module-03-classic-simulation/part-02-unity-and-assets/urdf-sdf',
            'module-03-classic-simulation/part-02-unity-and-assets/unity-intro',
            'module-03-classic-simulation/part-02-unity-and-assets/sensor-simulation'
          ],
        },
      ],
      collapsed: true, // All categories collapsed by default
    },
    {
      type: 'category',
      label: 'Module 4: NVIDIA Isaac - AI-Robot Brain',
      items: [
        {
          type: 'category',
          label: 'Part 1: Platform Basics',
          items: [
            'module-04-isaac-nvidia/part-01-platform-basics/isaac-overview',
            'module-04-isaac-nvidia/part-01-platform-basics/isaac-sim',
            'module-04-isaac-nvidia/part-01-platform-basics/isaac-ros'
          ],
        },
        {
          type: 'category',
          label: 'Part 2: Advanced Intelligence',
          items: [
            'module-04-isaac-nvidia/part-02-advanced-intelligence/vslam-navigation',
            'module-04-isaac-nvidia/part-02-advanced-intelligence/reinforcement-learning',
            'module-04-isaac-nvidia/part-02-advanced-intelligence/sim-to-real',
            'module-04-isaac-nvidia/part-02-advanced-intelligence/sim-integration'
          ],
        },
      ],
      collapsed: true, // All categories collapsed by default
    },
    {
      type: 'category',
      label: 'Module 5: Humanoid Control',
      items: [
        {
          type: 'category',
          label: 'Part 1: Locomotion',
          items: [
            'module-05-humanoid-control/part-01-locomotion/humanoid-kinematics',
            'module-05-humanoid-control/part-01-locomotion/bipedal-locomotion',
            'module-05-humanoid-control/part-01-locomotion/humanoid-overview'
          ],
        },
        {
          type: 'category',
          label: 'Part 2: Interaction',
          items: [
            'module-05-humanoid-control/part-02-interaction/manipulation',
            'module-05-humanoid-control/part-02-interaction/hri-design'
          ],
        },
      ],
      collapsed: true, // All categories collapsed by default
    },
    {
      type: 'category',
      label: 'Module 6: Cognitive AI',
      items: [
        {
          type: 'category',
          label: 'Part 1: NLP and Voice',
          items: [
            'module-06-cognitive-ai/part-01-nlp-and-voice/nlp-basics',
            'module-06-cognitive-ai/part-01-nlp-and-voice/voice-processing',
            'module-06-cognitive-ai/part-01-nlp-and-voice/conversational-robotics'
          ],
        },
        {
          type: 'category',
          label: 'Part 2: Integration',
          items: [
            'module-06-cognitive-ai/part-02-integration/gpt-integration',
            'module-06-cognitive-ai/part-02-integration/multimodal-interaction',
            'module-06-cognitive-ai/part-02-integration/capstone-project'
          ],
        },
      ],
      collapsed: true, // All categories collapsed by default
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        {
          type: 'category',
          label: 'Appendix A: Hardware',
          items: [
            'appendix-a-hardware/workstation-requirements',
            'appendix-a-hardware/edge-kit',
            'appendix-a-hardware/robot-platforms',
            'appendix-a-hardware/cloud-alternatives'
          ],
        },
        {
          type: 'category',
          label: 'Appendix B: Setup',
          items: [
            'appendix-b-setup/software-installation',
            'appendix-b-setup/ros2-setup',
            'appendix-b-setup/gazebo-setup',
            'appendix-b-setup/isaac-setup',
            'appendix-b-setup/troubleshooting'
          ],
        },
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Resources',
      items: [
        'resources/student-profiles',
        'resources/assessment-guidelines',
        'resources/code-example-validation',
        'resources/glossary',
        'resources/references',
        'resources/further-reading',
        'resources/community',
        'resources/api-contracts'
      ],
      collapsed: true,
    }
  ],
};

export default sidebars;