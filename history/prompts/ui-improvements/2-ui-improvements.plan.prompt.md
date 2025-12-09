----
id: 2
title: ui-improvements
stage: plan
date: 2025-12-09
model: Qwen
feature: ui-improvements
branch: 1-ui-improvements
user: Tayyaba Akbar
command: /sp.plan
labels: [ui, planning, architecture]
links:
  spec: specs/1-ui-improvements/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/1-ui-improvements/plan.md
  - specs/1-ui-improvements/research.md
  - specs/1-ui-improvements/data-model.md
  - specs/1-ui-improvements/quickstart.md
  - .qwen/agent-context.md
tests: []
prompt: |
  PROJECT: Physical AI & Humanoid Robotics Textbook - UI Implementation Plan
  CHANGE 1: HERO SECTION REDESIGN
  Files to Modify:
  - src/pages/index.tsx (or index.js)
  - src/pages/index.module.css
  - src/css/custom.css

  CHANGE 2: GLOWING MODULE CARDS
  Files to Modify:
  - src/components/HomepageFeatures/index.tsx (or wherever modules are)
  - src/components/HomepageFeatures/styles.module.css
  - Replace 6 module images in static/img/

  Assets Needed:
  - 6 custom module images (provided by user)
  - Place in: static/img/module-1.png through module-6.png

  IMPLEMENTATION ORDER:

  Step 1: Prepare Assets
  - Get robot hero image from user
  - Get 6 module images from user
  - Place in static/img/ folder
  - Optimize images (WebP format, < 200KB each)

  Step 2: Hero Section (Change 1)
  - Modify src/pages/index.tsx
  - Update src/pages/index.module.css
  - Test responsive behavior
  - Verify on mobile/tablet

  Step 3: Module Cards (Change 2)
  - Update HomepageFeatures component
  - Add glowing card CSS
  - Replace image paths
  - Test hover effects
  - Verify grid responsiveness

  Step 4: Sidebar Collapse (Change 3)
  - Edit sidebars.js
  - Add collapsed: true to all categories
  - Test navigation behavior
  - Verify intro doc opens by default

  Step 5: Testing & Polish
  - Cross-browser testing
  - Mobile responsiveness
  - Accessibility check
  - Performance audit
  - Build and deploy

  DESIGN TOKENS:

  Colors:
  - Black background: #000000
  - Card background: #1a1a1a
  - Purple glow: #8b5cf6
  - Purple light: #a78bfa
  - Text white: #ffffff
  - Text gray: #b8b8d1
  - Dark gradient: #1a1a2e to #16213e

  Spacing:
  - Card padding: 2rem
  - Grid gap: 2rem
  - Section padding: 4rem 2rem

  Border Radius:
  - Cards: 12px
  - Button: 8px
  - Images: 8px

  ACCESSIBILITY CONSIDERATIONS:
  - Maintain contrast ratio > 4.5:1
  - Add aria-labels to interactive elements
  - Keyboard navigation for cards
  - Alt text for all images
  - Focus indicators on buttons

  END PLAN
response: |
  Successfully created implementation plan for UI Improvements to Physical AI & Humanoid Robotics Textbook. Generated plan.md with technical context, constitution check, and project structure. Also created research.md with decisions and rationale, data-model.md with UI-related entities, quickstart.md with setup instructions, and updated agent context file. The plan covers all three changes: Hero Section Redesign, Module Cards Enhancement, and Sidebar Collapse Behavior.
outcome: |
  Implementation plan created successfully with branch '1-ui-improvements'.
  Created files:
  - specs/1-ui-improvements/plan.md
  - specs/1-ui-improvements/research.md
  - specs/1-ui-improvements/data-model.md
  - specs/1-ui-improvements/quickstart.md
  - .qwen/agent-context.md
  Plan includes technical context, research findings, data models, and quickstart guide to guide implementation.
---