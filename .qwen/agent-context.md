# Qwen Agent Context: Physical AI & Humanoid Robotics Textbook UI Improvements

## Project Overview
This project involves UI improvements for the Physical AI & Humanoid Robotics Textbook website built with Docusaurus. The changes include:
- Hero section redesign with split layout
- Module card enhancements with glowing effects
- Sidebar collapse behavior updates

## Technology Stack
- Docusaurus framework
- React components
- TypeScript/JavaScript
- CSS modules
- Node.js 18+

## Key Implementation Details

### 1. Hero Section Redesign
- Split layout: 70% right / 30% left
- Left side: Custom robot image
- Right side: Title, subtitle, and "Start Reading" button
- Black button with glowing purple border effect
- Modern, clean asymmetric layout

### 2. Module Cards Enhancement
- Black background with glowing purple border/shadow effect
- Consistent card dimensions
- Custom module images replacing existing ones
- Hover effects that intensify the glow
- Grid layout maintained for 6 modules

### 3. Sidebar Collapse Behavior
- All module categories collapsed by default
- Only "Introduction" document visible/open initially
- Users can click to expand specific modules as needed
- Clean, minimal initial view with progressive disclosure

### Design Tokens
- Colors:
  - Black background: #000000
  - Card background: #1a1a1a
  - Purple glow: #8b5cf6
  - Purple light: #a78bfa
  - Text white: #ffffff
  - Text gray: #b8b8d1
  - Dark gradient: #1a1a2e to #16213e
- Spacing:
  - Card padding: 2rem
  - Grid gap: 2rem
  - Section padding: 4rem 2rem
- Border Radius:
  - Cards: 12px
  - Button: 8px
  - Images: 8px

### Accessibility Considerations
- Maintain contrast ratio > 4.5:1
- Add aria-labels to interactive elements
- Keyboard navigation for cards
- Alt text for all images
- Focus indicators on buttons

## File Structure
- `/src/pages/index.tsx` - Homepage with hero section
- `/src/pages/index.module.css` - Hero section styles
- `/src/components/HomepageFeatures/index.tsx` - Module cards component
- `/src/components/HomepageFeatures/styles.module.css` - Module card styles
- `/src/css/custom.css` - Custom global styles
- `/static/img/` - Custom images (hero-robot.png, module-1.png through module-6.png)
- `/sidebars.js` - Sidebar configuration

## Implementation Order
1. Prepare Assets (get images and place in static/img/)
2. Hero Section (Change 1)
3. Module Cards (Change 2)
4. Sidebar Collapse (Change 3)
5. Testing & Polish

This agent context should be used when implementing the UI improvements for the Physical AI & Humanoid Robotics Textbook website.