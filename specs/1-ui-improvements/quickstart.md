# Quickstart Guide: UI Improvements for Physical AI & Humanoid Robotics Textbook

## Overview
This guide provides a quick setup and implementation path for the UI improvements to the Physical AI & Humanoid Robotics Textbook website.

## Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Git for version control
- A code editor (VS Code recommended)

## Setup Environment

1. Clone the repository (if not already done):
```bash
git clone <repository-url>
cd PHYSICAL_AI_BOOK
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Verify the setup by running the development server:
```bash
npm run start
# or
yarn start
```

## Implementation Steps

### Step 1: Prepare Assets
1. Obtain the robot hero image from the user and place in `static/img/hero-robot.png`
2. Obtain 6 custom module images and place in `static/img/` with names:
   - `module-1.png`, `module-2.png`, ..., `module-6.png`
3. Optimize images to WebP format if possible, keeping under 200KB each

### Step 2: Implement Hero Section Redesign
1. Edit `src/pages/index.tsx`:
   - Update the hero section to use a split layout (70% right / 30% left)
   - Add image container on the left with the robot image
   - Add content container on the right with title, subtitle, and button
   - Apply appropriate CSS classes

2. Update `src/pages/index.module.css`:
   - Create CSS rules for the split layout using Grid or Flexbox
   - Add styles for the title, subtitle, and "Start Reading" button
   - Implement the black background with purple glow effect for the button

3. Add custom styles to `src/css/custom.css` if needed for additional styling

### Step 3: Implement Module Cards Enhancement
1. Edit the HomepageFeatures component (likely in `src/components/HomepageFeatures/index.tsx`):
   - Modify the card rendering to include black background
   - Add CSS classes for the glowing purple border effect
   - Replace existing images with the new custom images
   - Implement hover effects if desired

2. Update the component's CSS file (e.g., `src/components/HomepageFeatures/styles.module.css`):
   - Add styles for the card background (#000000)
   - Add styles for the glowing border/shadow effect (#8b5cf6)
   - Implement consistent card dimensions
   - Add hover states to intensify glow

### Step 4: Implement Sidebar Collapse Behavior
1. Edit the `sidebars.js` file:
   - Find the sidebar configuration for the textbook
   - Add `collapsed: true` to all module categories
   - Ensure the "Introduction" document is set to be visible/open initially

2. If needed, add custom JavaScript to handle the default collapse behavior

### Step 5: Testing & Polish
1. Cross-browser testing:
   - Test in Chrome, Firefox, Safari, and Edge
   - Verify that all UI elements display correctly in each browser

2. Mobile responsiveness:
   - Use browser dev tools to test on various screen sizes
   - Ensure the split layout works well on mobile
   - Check that the sidebar behaves correctly on mobile

3. Accessibility check:
   - Verify all interactive elements are keyboard accessible
   - Check that contrast ratios are >4.5:1
   - Add appropriate alt text to all images
   - Add aria-labels where needed

4. Performance audit:
   - Check that page load times haven't increased significantly
   - Verify that images are optimized
   - Test that animations/glow effects don't impact performance

5. Build and verify:
   ```bash
   npm run build
   npm run serve
   ```
   - Test the built version locally before deploying

## Key CSS Classes and Variables

### Colors
- Black background: `#000000`
- Card background: `#1a1a1a`
- Purple glow: `#8b5cf6`
- Purple light: `#a78bfa`
- Text white: `#ffffff`
- Text gray: `#b8b8d1`
- Dark gradient: `#1a1a2e to #16213e`

### Spacing
- Card padding: `2rem`
- Grid gap: `2rem`
- Section padding: `4rem 2rem`

### Border Radius
- Cards: `12px`
- Button: `8px`
- Images: `8px`

### Glow Effect CSS
```css
.glowing-element {
  border: 2px solid #8b5cf6;
  box-shadow: 0 0 15px #8b5cf6;
  transition: box-shadow 0.3s ease;
}

.glowing-element:hover {
  box-shadow: 0 0 25px #8b5cf6;
}
```

## Common Issues and Solutions

1. **Responsive Layout Issues**: Ensure you're using relative units (%, rem, em) instead of fixed pixels where appropriate. Use media queries to adjust layouts for different screen sizes.

2. **Image Loading**: If images are slow to load, ensure they're properly optimized and consider implementing lazy loading for off-screen images.

3. **Performance**: If glow effects or animations are causing performance issues, consider using `will-change` property or `transform` and `opacity` for animations.

4. **Sidebar Not Collapsing**: Verify that the Docusaurus sidebar configuration is correctly set with `collapsed: true` properties.