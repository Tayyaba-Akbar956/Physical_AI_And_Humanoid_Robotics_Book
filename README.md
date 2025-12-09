# Physical AI & Humanoid Robotics Textbook

This repository contains a comprehensive educational textbook that teaches students how to design, simulate, and deploy humanoid robots that can interact naturally with the physical world. The book transforms abstract AI concepts into tangible robotic applications, bridging the gap between digital intelligence and physical embodiment.

## About

This textbook follows a 13-week course structure covering:
- **Module 1**: Foundations - Introduction to Physical AI concepts
- **Module 2**: ROS 2 - The robotic nervous system
- **Module 3**: Simulation - Creating digital twins with Gazebo and Unity
- **Module 4**: NVIDIA Isaac - Adding AI to robots
- **Module 5**: Humanoid Control - Locomotion and interaction
- **Module 6**: Cognitive AI - Conversational robotics

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

This will start a local development server and open the textbook in your default browser at `http://localhost:3000`. Most changes are live-reloaded automatically.

## Building

To build the site for production:

```bash
npm run build
```

This generates optimized static files in the `build/` directory, ready for deployment.

## Contributing

We welcome contributions to enhance this educational resource! Here's how you can contribute:

### Content Creation
1. Create new MDX files in the appropriate module directory in `docs/`
2. Update `sidebars.js` to add the new content to navigation
3. Ensure all content follows the required chapter structure:
   - Chapter Title
   - Learning Objectives (3-5 bullet points)
   - Introduction (why this matters)
   - Core Concepts (theory)
   - Practical Implementation (code/examples)
   - Hands-on Exercise
   - Key Takeaways
   - Further Reading
   - Next Steps (link to next chapter)

### Development Setup
1. Fork and clone the repository
2. Install dependencies with `npm install`
3. Start development server with `npm start`
4. Make your changes
5. Test by building with `npm run build`
6. Submit a pull request

### Technical Requirements
- All code examples must be tested and functional
- Content should be accessible to students with basic AI knowledge
- Follow the established color scheme and design guidelines
- Ensure responsive design works on mobile and desktop

### Module Structure
The textbook is organized into 6 modules following a 13-week course structure:
- Module 1: Foundations of Physical AI (Weeks 1-2)
- Module 2: ROS 2 - The Robotic Nervous System (Weeks 3-5)
- Module 3: Simulation - Digital Twin Environments (Weeks 6-7)
- Module 4: NVIDIA Isaac - AI-Robot Brain (Weeks 8-10)
- Module 5: Humanoid Control (Weeks 11-12)
- Module 6: Cognitive AI - Capstone Project (Week 13)

### Code of Conduct
Please follow our community guidelines to ensure a welcoming environment for all contributors.

## License

This project is licensed under the MIT License - see the LICENSE file for details.