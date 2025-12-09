----
id: 3
title: ui-improvements
stage: tasks
date: 2025-12-09
model: Qwen
feature: ui-improvements
branch: 1-ui-improvements
user: Tayyaba Akbar
command: /sp.tasks
labels: [ui, tasks, implementation]
links:
  spec: specs/1-ui-improvements/spec.md
  plan: specs/1-ui-improvements/plan.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/1-ui-improvements/tasks.md
tests: []
prompt: |
  PROJECT: Physical AI & Humanoid Robotics Textbook - UI Implementation Tasks
  TASKS:
  TASK 1: HERO SECTION REDESIGN
  PRIORITY: P1

  Files to modify:
  - src/pages/index.tsx
  - src/pages/index.module.css

  Requirements:
  - Split layout: 70% right / 30% left
  - Left side: Custom robot image
  - Right side: Title, subtitle, button with styling
  - Black button with glowing purple border
  - Responsive design

  TASK 2: MODULE CARDS ENHANCEMENT
  PRIORITY: P1

  Files to modify:
  - src/components/HomepageFeatures/index.tsx
  - src/components/HomepageFeatures/styles.module.css
  - static/img/ (module images)

  Requirements:
  - Replace images with custom ones
  - Add black background to cards
  - Add glowing purple border effect
  - Implement hover effects
  - Maintain grid layout

  TASK 3: SIDEBAR COLLAPSE BEHAVIOR
  PRIORITY: P2

  Files to modify:
  - sidebars.js
  - potentially src/theme/ components

  Requirements:
  - All categories collapsed by default
  - Only "Introduction" visible initially
  - Click to expand functionality

  TASK 4: ACCESSIBILITY & RESPONSIVENESS
  PRIORITY: P2

  Files to modify:
  - All UI components modified in previous tasks
  - src/css/custom.css (if needed)

  Requirements:
  - Alt text for images
  - ARIA attributes where needed
  - Keyboard navigation
  - Responsive design across devices
  - Contrast ratio >4.5:1
  - Focus indicators

  END TASKS
response: |
  Successfully generated implementation tasks for UI Improvements to Physical AI & Humanoid Robotics Textbook. Created tasks.md with 44 tasks organized by user story and priority. Tasks are organized in phases: Setup, Foundational, US1 (Hero Section), US2 (Module Cards), US3 (Sidebar), and Polish. Each task follows the checklist format with proper IDs and story labels. The tasks cover all requirements from the specification and plan.
outcome: |
  Implementation tasks created successfully with branch '1-ui-improvements'.
  Created files:
  - specs/1-ui-improvements/tasks.md
  Tasks organized in phases with 44 total tasks, including setup, foundational, user stories (US1-3), and polish tasks. Each task follows checklist format with sequential IDs and story labels. Tasks are prioritized and include acceptance criteria.
---