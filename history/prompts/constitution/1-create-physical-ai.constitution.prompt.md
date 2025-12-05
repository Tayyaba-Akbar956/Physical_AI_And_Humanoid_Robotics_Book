----
id: 1
title: Create Physical AI Constitution
stage: constitution
date: 2025-12-05
surface: agent
model: gpt-4o
feature: none
branch: main
user: Tayyaba
command: /sp.constitution PROJECT: Physical AI & Humanoid Robotics Textbook
labels: [constitution, textbook, ai, robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

/sp.constitution 
PROJECT: Physical AI & Humanoid Robotics Textbook

CORE MISSION:
Create a comprehensive, educational textbook for teaching Physical AI and Humanoid Robotics using Docusaurus. The book must bridge the gap between digital AI and physical embodied intelligence, covering ROS 2, simulation tools, NVIDIA Isaac platform, and conversational robotics.

NON-NEGOTIABLE PRINCIPLES:

1. EDUCATIONAL CLARITY
   - Every concept must be explained from first principles
   - Use progressive complexity: start simple, build to advanced
   - Include real-world examples and practical applications
   - Avoid jargon without explanation
   - Each chapter must have clear learning objectives

2. TECHNICAL ACCURACY
   - All code examples must be functional and tested
   - Hardware specifications must be current and accurate
   - Software versions must be specified (ROS 2 Humble/Iron, Ubuntu 22.04)
   - Citations for technical claims and specifications
   - No outdated or deprecated methods

3. STRUCTURED LEARNING PATH
   - Follow the 13-week course outline strictly
   - Each module builds on previous knowledge
   - Maintain logical progression: ROS 2 → Simulation → Isaac → Humanoids → Conversational AI
   - Prerequisites clearly stated for each chapter
   - Consistent chapter structure across all modules

4. PRACTICAL ORIENTATION
   - Theory balanced with hands-on examples
   - Include code snippets with explanations
   - Provide step-by-step tutorials
   - Real-world use cases and applications
   - Project-based learning approach

5. ACCESSIBILITY
   - Content suitable for students with basic AI knowledge
   - Accommodate different hardware setups (workstation vs cloud)
   - Clear installation and setup instructions
   - Troubleshooting guidance
   - Alternative approaches for budget constraints

6. COMPREHENSIVE COVERAGE
   - All 4 modules must be fully covered:
     * Module 1: ROS 2 (Robotic Nervous System)
     * Module 2: Gazebo & Unity (Digital Twin)
     * Module 3: NVIDIA Isaac (AI-Robot Brain)
     * Module 4: Vision-Language-Action (VLA)
   - Hardware requirements section included
   - Assessment guidelines for each module

7. PROFESSIONAL QUALITY
   - Proper grammar and technical writing standards
   - Consistent terminology throughout
   - Professional diagrams and illustrations where needed
   - Well-formatted code blocks with syntax highlighting
   - Clean, readable Markdown

8. DOCUSAURUS STANDARDS
   - Proper sidebar navigation structure
   - Responsive design
   - Fast loading times
   - Mobile-friendly layout
   - Search functionality enabled

9. GITHUB READINESS
   - Well-documented README
   - Clear repository structure
   - Proper .gitignore configuration
   - MIT or appropriate open-source license
   - Contribution guidelines (if applicable)

10. DEPLOYMENT REQUIREMENTS
    - Must successfully build with `npm run build`
    - No broken links or missing resources
    - Deployed version must be publicly accessible
    - All images and assets properly referenced
    - Cross-browser compatibility

CONTENT STRUCTURE RULES:

Chapter Template (Mandatory for all chapters):
1. Chapter Title
2. Learning Objectives (3-5 bullet points)
3. Introduction (why this matters)
4. Core Concepts (theory)
5. Practical Implementation (code/examples)
6. Hands-on Exercise
7. Key Takeaways
8. Further Reading
9. Next Steps (link to next chapter)

TECHNICAL STACK (Fixed):
- Docusaurus (latest stable version)
- Node.js 18+
- React components for interactive elements
- Markdown/MDX for content
- GitHub Pages or Vercel for hosting

FORBIDDEN:
- No incomplete or "TODO" sections in final version
- No broken code examples
- No unsupported hardware recommendations
- No contradictory information between chapters
- No plagiarized content
- No overly simplified explanations that mislead
- No outdated library versions without migration notes

SUCCESS CRITERIA:
- Complete 13-chapter structure
- All code examples verified
- Proper deployment to GitHub Pages/Vercel
- Professional documentation
- Clear navigation and user experience
- Passes all Docusaurus build checks
- Demo video ready (under 90 seconds)

TARGET AUDIENCE:
Students with basic AI/programming knowledge who want to learn Physical AI, robotics simulation, and humanoid robot development. They should be able to follow along and build practical skills.

TONE AND STYLE:
- Educational but engaging
- Technical but accessible
- Encouraging and supportive
- Professional yet conversational
- Focus on empowering learners

## Response snapshot

I've successfully created the project constitution for the Physical AI & Humanoid Robotics Textbook based on your specifications. The constitution file has been saved to `.specify/memory/constitution.md` with the following details:

**New Version**: 1.0.0 (Initial version)
**Bump Rationale**: This is the initial creation of the constitution based on your requirements for the Physical AI & Humanoid Robotics textbook project.

The constitution includes all 10 non-negotiable principles you specified:
1. Educational Clarity
2. Technical Accuracy
3. Structured Learning Path
4. Practical Orientation
5. Accessibility
6. Comprehensive Coverage
7. Professional Quality
8. Docusaurus Standards
9. GitHub Readiness
10. Deployment Requirements

Additional sections were added as specified:
- Content Structure Rules with mandatory chapter template
- Technical Stack requirements
- Forbidden Practices
- Success Criteria
- Target Audience guidelines
- Tone and Style specifications
- Governance section

The file includes an HTML comment at the top with the Sync Impact Report showing what changes were made.

The dependent templates were checked for consistency, and no updates were required to the existing template files as they are generic and adaptable.

The constitution follows all your formatting requirements with proper headings, spacing, and structure.

## Outcome

- ✅ Impact: Created the complete project constitution for the Physical AI & Humanoid Robotics textbook project
- 🧪 Tests: No tests required for constitution creation
- 📁 Files: Created .specify/memory/constitution.md with all 10 principles and additional sections
- 🔁 Next prompts: Ready to proceed with spec creation for individual chapters/modules
- 🧠 Reflection: The constitution successfully captures all specified principles and requirements for the textbook project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin creating specifications for individual chapters based on the constitution