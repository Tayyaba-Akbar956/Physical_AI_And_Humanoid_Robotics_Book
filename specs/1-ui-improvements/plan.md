# Implementation Plan: UI Improvements for Physical AI & Humanoid Robotics Textbook

**Branch**: `1-ui-improvements` | **Date**: 2025-12-09 | **Spec**: [link](./spec.md)
**Input**: Feature specification from `/specs/1-ui-improvements/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement UI improvements to the Physical AI & Humanoid Robotics Textbook website based on the feature specification. This includes redesigning the hero section with split layout and custom robot image, enhancing module cards with black backgrounds and glowing purple borders, and modifying the sidebar to be collapsed by default. The implementation will follow Docusaurus standards and maintain responsive design across all devices.

## Technical Context

**Language/Version**: TypeScript/JavaScript ES6+ (as per Docusaurus requirements)
**Primary Dependencies**: Docusaurus framework, React components, CSS modules, Node.js 18+
**Storage**: N/A (frontend-only UI changes)
**Testing**: Jest for unit tests, Cypress for end-to-end tests (as per Docusaurus standards)
**Target Platform**: Web-based (cross-browser compatible)
**Project Type**: Web (Docusaurus documentation site)
**Performance Goals**: Maintain fast loading times, ensure no performance degradation from visual enhancements
**Constraints**: Responsive design must work on mobile, tablet, and desktop; maintain accessibility standards; ensure all UI elements are keyboard navigable
**Scale/Scope**: Applies to the entire Physical AI textbook website, primarily the homepage and navigation

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the constitution file, the following gates apply:
- EDUCATIONAL CLARITY: UI enhancements should improve how content is presented and accessed
- TECHNICAL ACCURACY: CSS/JS implementations must follow current standards
- STRUCTURED LEARNING PATH: Navigation improvements should maintain logical content structure
- PRACTICAL ORIENTATION: Visual enhancements should improve user engagement
- ACCESSIBILITY: All UI changes must maintain or improve accessibility compliance
- PROFESSIONAL QUALITY: Visual design must be polished and consistent
- DOCUSAURUS STANDARDS: Implementation must follow Docusaurus patterns
- GITHUB READINESS: Changes must build correctly with Docusaurus
- DEPLOYMENT REQUIREMENTS: Site must build successfully after changes

## Project Structure

### Documentation (this feature)

```text
specs/1-ui-improvements/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
src/
├── pages/
│   └── index.tsx        # Homepage with hero section
├── components/
│   └── HomepageFeatures/ # Module cards component
├── css/
│   └── custom.css       # Custom styles
└── theme/
    └── MDXContent/      # Custom theme components if needed

static/
└── img/                 # Custom images (robot image, module images)
```

**Structure Decision**: Single web application following Docusaurus structure. The UI changes are contained within the existing Docusaurus project without requiring any additional services or complex architecture.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|