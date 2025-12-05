<!-- 
SYNC IMPACT REPORT:
Version change: N/A (Initial constitution) → 1.0.0
Modified principles: N/A (New principles added)
Added sections: All 10 principles and additional sections
Removed sections: N/A
Templates requiring updates: ✅ plan-template.md (updated) / ⚠ tasks-template.md (may need task-specific updates) / ⚠ spec-template.md (may need content-specific updates)
Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics Textbook Constitution

## Core Principles

### EDUCATIONAL CLARITY
Every concept must be explained from first principles. Use progressive complexity: start simple, build to advanced. Include real-world examples and practical applications. Avoid jargon without explanation. Each chapter must have clear learning objectives.

### TECHNICAL ACCURACY
All code examples must be functional and tested. Hardware specifications must be current and accurate. Software versions must be specified (ROS 2 Humble/Iron, Ubuntu 22.04). Citations for technical claims and specifications. No outdated or deprecated methods.

### STRUCTURED LEARNING PATH
Follow the 13-week course outline strictly. Each module builds on previous knowledge. Maintain logical progression: ROS 2 → Simulation → Isaac → Humanoids → Conversational AI. Prerequisites clearly stated for each chapter. Consistent chapter structure across all modules.

### PRACTICAL ORIENTATION
Theory balanced with hands-on examples. Include code snippets with explanations. Provide step-by-step tutorials. Real-world use cases and applications. Project-based learning approach.

### ACCESSIBILITY
Content suitable for students with basic AI knowledge. Accommodate different hardware setups (workstation vs cloud). Clear installation and setup instructions. Troubleshooting guidance. Alternative approaches for budget constraints.

### COMPREHENSIVE COVERAGE
All 4 modules must be fully covered: Module 1: ROS 2 (Robotic Nervous System), Module 2: Gazebo & Unity (Digital Twin), Module 3: NVIDIA Isaac (AI-Robot Brain), Module 4: Vision-Language-Action (VLA). Hardware requirements section included. Assessment guidelines for each module.

### PROFESSIONAL QUALITY
Proper grammar and technical writing standards. Consistent terminology throughout. Professional diagrams and illustrations where needed. Well-formatted code blocks with syntax highlighting. Clean, readable Markdown.

### DOCUSAURUS STANDARDS
Proper sidebar navigation structure. Responsive design. Fast loading times. Mobile-friendly layout. Search functionality enabled.

### GITHUB READINESS
Well-documented README. Clear repository structure. Proper .gitignore configuration. MIT or appropriate open-source license. Contribution guidelines (if applicable).

### DEPLOYMENT REQUIREMENTS
Must successfully build with `npm run build`. No broken links or missing resources. Deployed version must be publicly accessible. All images and assets properly referenced. Cross-browser compatibility.

## Content Structure Rules

### Chapter Template (Mandatory for all chapters)
1. Chapter Title
2. Learning Objectives (3-5 bullet points)
3. Introduction (why this matters)
4. Core Concepts (theory)
5. Practical Implementation (code/examples)
6. Hands-on Exercise
7. Key Takeaways
8. Further Reading
9. Next Steps (link to next chapter)

## Technical Stack
- Docusaurus (latest stable version)
- Node.js 18+
- React components for interactive elements
- Markdown/MDX for content
- GitHub Pages or Vercel for hosting

## Forbidden Practices
- No incomplete or "TODO" sections in final version
- No broken code examples
- No unsupported hardware recommendations
- No contradictory information between chapters
- No plagiarized content
- No overly simplified explanations that mislead
- No outdated library versions without migration notes

## Success Criteria
- Complete 13-chapter structure
- All code examples verified
- Proper deployment to GitHub Pages/Vercel
- Professional documentation
- Clear navigation and user experience
- Passes all Docusaurus build checks
- Demo video ready (under 90 seconds)

## Target Audience
Students with basic AI/programming knowledge who want to learn Physical AI, robotics simulation, and humanoid robot development. They should be able to follow along and build practical skills.

## Tone and Style
Educational but engaging. Technical but accessible. Encouraging and supportive. Professional yet conversational. Focus on empowering learners.

## Governance
This constitution supersedes all other development practices. All pull requests and reviews must verify compliance with these principles. Major changes to the constitution require proper documentation and approval. The implementation plan must align with these principles, and any deviations must be justified. Compliance reviews are expected at key milestones in the development process.

**Version**: 1.0.0 | **Ratified**: 2025-06-13 | **Last Amended**: 2025-12-05