# Data Model: Physical AI & Humanoid Robotics Textbook

## Learning Modules
- **Name**: Module
- **Fields**:
  - id: string (unique identifier, e.g., "module-01-foundations")
  - title: string (e.g., "Foundations of Physical AI")
  - description: string (brief overview of module content)
  - duration: number (estimated weeks to complete)
  - prerequisites: string[] (IDs of required modules)
  - objectives: string[] (learning objectives)
  - parts: Part[] (sub-components of the module)

## Parts
- **Name**: Part
- **Fields**:
  - id: string (unique identifier, e.g., "part-01-concepts")
  - title: string (e.g., "Core Concepts")
  - description: string (brief overview of part content)
  - chapters: Chapter[] (collection of chapters in this part)

## Chapters
- **Name**: Chapter
- **Fields**:
  - id: string (unique identifier, e.g., "01-intro-physical-ai")
  - title: string (chapter title)
  - content: string (the actual chapter content in MDX format)
  - objectives: string[] (3-5 learning objectives)
  - prerequisites: string[] (what students should know before reading)
  - nextSteps: string[] (links to subsequent chapters/modules)
  - codeExamples: CodeExample[] (list of code examples in the chapter)
  - exercises: Exercise[] (practical exercises for students)

## Code Examples
- **Name**: CodeExample
- **Fields**:
  - id: string (unique identifier)
  - title: string (e.g., "ROS 2 Publisher Node")
  - description: string (what the example demonstrates)
  - code: string (the actual code snippet)
  - language: string (e.g., "python", "cpp", "bash")
  - explanation: string (step-by-step explanation of the code)

## Exercises
- **Name**: Exercise
- **Fields**:
  - id: string (unique identifier)
  - title: string (exercise title)
  - description: string (what students need to do)
  - difficulty: "beginner" | "intermediate" | "advanced"
  - estimatedTime: number (time in minutes to complete)
  - solution: string (optional solution or guidance)

## Humanoid Robot Models
- **Name**: RobotModel
- **Fields**:
  - id: string (e.g., "humanoid-simulation-1")
  - name: string (display name for the robot)
  - urdfPath: string (path to URDF file for ROS 2)
  - description: string (what this model demonstrates)
  - capabilities: string[] (e.g., ["walking", "object_manipulation", "voice_interaction"])
  - simulator: "gazebo" | "isaac_sim" | "unity" (which simulator this model is for)

## Simulation Environments
- **Name**: SimulationEnvironment
- **Fields**:
  - id: string (e.g., "basic-warehouse-env")
  - name: string (display name)
  - description: string (what the environment is for)
  - simulator: "gazebo" | "isaac_sim" | "unity"
  - robotModels: RobotModel[] (compatible robot models)
  - tasks: string[] (what tasks can be performed in this environment)
  - configurationFiles: string[] (paths to configuration files)

## Assessment Criteria
- **Name**: Assessment
- **Fields**:
  - id: string (unique identifier)
  - title: string (e.g., "ROS 2 Package Development")
  - description: string (what skill is being assessed)
  - module: string (which module this assessment belongs to)
  - type: "practical" | "theoretical" | "project" (type of assessment)
  - criteria: string[] (what will be evaluated)
  - rubric: string (grading guidelines)
  - sampleSolution: string (example of expected outcome)

## Hardware Specifications
- **Name**: HardwareSpec
- **Fields**:
  - id: string (e.g., "workstation-minimum")
  - name: string (e.g., "Minimum Workstation Requirements")
  - category: "workstation" | "edge_device" | "robot_platform" | "cloud_alternative"
  - specifications: string[] (list of hardware requirements)
  - costEstimate: string (approximate cost or "Free" for cloud options)
  - alternatives: string[] (alternative options with different specs/costs)
  - compatibility: string[] (which tools/components are compatible)

## User (Student) Profiles
- **Name**: StudentProfile
- **Fields**:
  - id: string (e.g., "beginner-ai-student")
  - name: string (e.g., "Beginner AI Student")
  - background: string (what knowledge they're expected to have)
  - goals: string[] (what they want to achieve)
  - challenges: string[] (potential obstacles they might face)
  - recommendedPath: string (suggested learning path through the textbook)

## Validation Rules
- Each Module must have at least one Part
- Each Part must have at least one Chapter
- Each Chapter must have 3-5 learning objectives
- Each CodeExample must have a language specified
- Each Exercise must have a difficulty level
- Each Assessment must belong to a specific module
- All IDs must follow the format: "category-nn-description" (e.g., "module-01-foundations")
- All content files must be in MDX format

## State Transitions
- Chapter: draft → review → approved → published
- Module: planned → in-development → review → approved → published
- Assessment: created → reviewed → validated → active
- RobotModel: designed → tested → validated → documented → published