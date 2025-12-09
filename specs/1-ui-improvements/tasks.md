# Implementation Tasks: UI Improvements for Physical AI & Humanoid Robotics Textbook

**Branch**: `1-ui-improvements` | **Date**: 2025-12-09 | **Spec**: [link](./spec.md) | **Plan**: [link](./plan.md)

**Input**: Feature specification from `/specs/1-ui-improvements/spec.md` and implementation plan from `/specs/1-ui-improvements/plan.md`

**Note**: This template is filled in by the `/sp.tasks` command. See `.specify/templates/commands/tasks.md` for the execution workflow.

## Task Priorities

- **P1**: Critical path functionality - required for basic feature operation
- **P2**: Important functionality that enhances the feature
- **P3**: Nice-to-have or optimization tasks

## Phase 1: Setup

- [X] T001 Prepare image assets (robot hero image and 6 custom module images) and place in static/img/
- [X] T002 Verify development environment with Docusaurus (Node.js 18+, npm/yarn)

## Phase 2: Foundational

- [X] T003 Define CSS custom properties for design tokens (colors: #000000, #1a1a1a, #8b5cf6, #a78bfa, #ffffff, #b8b8d1; spacing: 2rem card padding, 2rem grid gap, 4rem 2rem section padding; border-radius: 12px cards, 8px button/images)
- [X] T004 Create reusable CSS class for glowing effect using box-shadow with purple color (#8b5cf6) and transition

## Phase 3: [US1] View Enhanced Hero Section (P1)

**Description**: Implement the redesigned hero section with split layout, custom robot image, and glowing button

**Tasks**:

- [X] T005 [US1] Update src/pages/index.tsx to implement split layout (70% right / 30% left) using CSS Grid or Flexbox
- [X] T006 [US1] Add container on left side for custom robot image in src/pages/index.tsx
- [X] T007 [US1] Add container on right side with title "Physical AI & Humanoid Robotics Textbook", subtitle "Bridging the gap between digital AI and physical embodiment", and button "Start Reading" in src/pages/index.tsx
- [X] T008 [US1] Update src/pages/index.module.css to style the split layout with appropriate dimensions
- [X] T009 [US1] Implement black "Start Reading" button with glowing purple border effect in src/pages/index.module.css
- [X] T010 [US1] Implement responsive design for hero section to work on mobile/tablet in src/pages/index.module.css
- [X] T011 [US1] Add alt text to the robot image for accessibility

**Acceptance Tests**:

- [X] T012 [US1] Verify that the hero section displays with 70% right / 30% left split layout
- [X] T013 [US1] Verify that the left side shows the custom robot image
- [X] T014 [US1] Verify that the right side shows the correct title, subtitle, and button
- [X] T015 [US1] Verify that the "Start Reading" button has black background with glowing purple border
- [X] T016 [US1] Verify that the layout is responsive and works on mobile devices

## Phase 4: [US2] View Enhanced Module Cards (P2)

**Description**: Transform module cards with black backgrounds, glowing purple borders, and custom images

**Tasks**:

- [X] T017 [US2] Replace existing module images with user-provided custom images in src/components/HomepageFeatures/index.tsx
- [X] T018 [US2] Update src/components/HomepageFeatures/index.tsx to implement card component structure
- [X] T019 [US2] Add black background to module cards in src/components/HomepageFeatures/styles.module.css
- [X] T020 [US2] Apply glowing purple border/shadow effect to module cards in src/components/HomepageFeatures/styles.module.css
- [X] T021 [US2] Ensure consistent card dimensions across all 6 modules in src/components/HomepageFeatures/styles.module.css
- [X] T022 [US2] Implement hover effect that intensifies the glow in src/components/HomepageFeatures/styles.module.css
- [X] T023 [US2] Maintain grid layout for 6 modules in src/components/HomepageFeatures/styles.module.css
- [X] T024 [US2] Add accessibility attributes to module cards (aria-labels, focus indicators)

**Acceptance Tests**:

- [X] T025 [US2] Verify that all 6 module cards have black background with glowing purple border
- [X] T026 [US2] Verify that all cards display custom images as specified
- [X] T027 [US2] Verify that all cards have consistent dimensions
- [X] T028 [US2] Verify that hover effect intensifies the glow on module cards
- [X] T029 [US2] Verify that cards maintain grid layout as expected

## Phase 5: [US3] Navigate Using Collapsible Sidebar (P3)

**Description**: Modify sidebar to have all categories collapsed by default with only "Introduction" visible

**Tasks**:

- [X] T030 [US3] Update sidebars.js to set all module categories to collapsed by default
- [X] T031 [US3] Ensure "Introduction" document is visible/open initially in sidebars.js
- [X] T032 [US3] Verify that users can click to expand specific modules as needed
- [X] T033 [US3] Test navigation behavior to ensure it works as expected after sidebar changes

**Acceptance Tests**:

- [X] T034 [US3] Verify that all module categories are collapsed by default in the sidebar
- [X] T035 [US3] Verify that only "Introduction" document is visible/open initially
- [X] T036 [US3] Verify that users can click to expand specific modules as needed

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T037 Implement accessibility features: ensure contrast ratio >4.5:1 for all text elements
- [X] T038 Add keyboard navigation support for cards and sidebar
- [X] T039 Implement focus indicators on interactive elements
- [X] T040 Verify responsive design across different screen sizes (mobile, tablet, desktop)
- [X] T041 Cross-browser testing (Chrome, Firefox, Safari, Edge)
- [X] T042 Performance optimization to ensure no degradation from visual enhancements
- [X] T043 Image optimization to ensure appropriate file sizes and formats
- [X] T044 Verify that the site builds successfully with `npm run build`

## Dependencies

- T005[T003] - Hero section implementation requires design tokens
- T017[T001] - Module cards require custom images
- T019[T003] - Module cards styling requires design tokens
- T020[T004] - Module cards glow effect requires glow utility class

## Parallel Execution Opportunities

- [P] T005, T017 - Hero section and module cards can be worked on in parallel after foundational tasks
- [P] T008, T019, T020 - Styling tasks can be parallelized after components are updated

## Implementation Strategy

1. Start with foundational setup and design tokens
2. Implement User Story 1 (Hero Section - P1) as MVP
3. Add User Story 2 (Module Cards - P2) 
4. Complete User Story 3 (Sidebar - P3)
5. Address all cross-cutting concerns and polish

## Definition of Done

- [ ] All P1 tasks completed and tested
- [ ] All P2 tasks completed and tested
- [ ] All P3 tasks completed and tested
- [ ] Code reviewed and approved
- [ ] All acceptance tests pass
- [ ] Performance requirements met
- [ ] Accessibility requirements met
- [ ] Responsive design verified