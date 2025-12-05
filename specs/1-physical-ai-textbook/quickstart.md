# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Prerequisites

Before starting with the Physical AI & Humanoid Robotics textbook, ensure you have:

- Basic knowledge of AI/programming concepts
- A computer running Windows, macOS, or Linux
- Node.js 18 or higher installed
- Git installed
- A modern web browser

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Tayyaba-Akbar956/Physical_AI_And_Humanoid_Robotics_Book.git
cd Physical_AI_And_Humanoid_Robotics_Book
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Run the Development Server

```bash
npm start
```

This command starts a local development server and opens the textbook in your default browser at `http://localhost:3000`. Most changes are live-reloaded automatically.

### 4. Explore the Content Structure

The textbook content is organized as follows:

- `docs/` - Contains all the textbook content in MDX format, organized by modules
- `src/` - Contains custom React components and pages
- `static/` - Contains static assets like images and files
- `docusaurus.config.js` - Main configuration file
- `sidebars.js` - Navigation structure

### 5. Navigate the Textbook

The textbook follows a 13-week course structure:

- **Module 1**: Foundations - Introduction to Physical AI concepts
- **Module 2**: ROS 2 - The robotic nervous system
- **Module 3**: Simulation - Creating digital twins with Gazebo and Unity
- **Module 4**: NVIDIA Isaac - Adding AI to robots
- **Module 5**: Humanoid Control - Locomotion and interaction
- **Module 6**: Cognitive AI - Conversational robotics

Each module contains multiple parts with chapters that follow the required structure:
1. Chapter Title
2. Learning Objectives (3-5 bullet points)
3. Introduction (why this matters)
4. Core Concepts (theory)
5. Practical Implementation (code/examples)
6. Hands-on Exercise
7. Key Takeaways
8. Further Reading
9. Next Steps (link to next chapter)

## Development Tasks

### Build the Static Site

To build the site for production:

```bash
npm run build
```

This generates optimized static files in the `build/` directory, ready for deployment.

### Deploy to GitHub Pages

```bash
npm run deploy
```

### Run Tests

```bash
npm run serve
```

This command serves the built site locally to test the production build before deployment.

## Contributing to the Textbook

### Adding New Content

1. Create new MDX files in the appropriate module directory in `docs/`
2. Update `sidebars.js` to add the new content to navigation
3. Ensure all content follows the required chapter structure

### Modifying Components

Custom React components are in `src/components/`. You can modify these to change the appearance and behavior of the textbook.

### Adding Assets

Images and other assets go in the `static/` directory. Reference them in MDX files using absolute paths like `/img/your-image.png`.

## Troubleshooting

### Common Issues

1. **Node.js version too old**: Ensure you're using Node.js 18 or higher
2. **Dependency installation fails**: Try clearing npm cache with `npm cache clean --force` and reinstalling
3. **Build fails**: Check that all links and image references are correct
4. **Development server not starting**: Verify all dependencies are installed and try restarting

### Getting Help

- Check the troubleshooting section in Appendix B of the textbook
- Review the GitHub issues for similar problems
- Contact the development team through the community section

## Hardware Requirements

For the hands-on exercises in the textbook, you may need:

- For simulation exercises: Workstation with at least 8GB RAM and a modern CPU
- For running ROS 2: Ubuntu 22.04, macOS 10.14+, or Windows with WSL2
- For advanced Isaac Sim exercises: GPU with at least 4GB VRAM (recommended: 8GB+)

Note: Many exercises can be completed using cloud alternatives if local hardware requirements aren't met.

## Support

- For technical issues with the textbook platform: Check the GitHub repository issues
- For content questions: Refer to the community resources section
- For installation help: See Appendix B Setup Guide in the textbook