# Data Model: UI Improvements for Physical AI & Humanoid Robotics Textbook

## Overview
This document describes the data models related to the UI improvements for the Physical AI & Humanoid Robotics Textbook website. While the feature is mainly UI-focused, there are some data structures that need to be understood for proper implementation.

## Entities

### 1. Hero Section Content
**Description**: The content and metadata for the hero section on the homepage

**Fields**:
- `title`: string - The main title text ("Physical AI & Humanoid Robotics Textbook")
- `subtitle`: string - The subtitle text ("Bridging the gap between digital AI and physical embodiment")
- `buttonText`: string - The text for the primary button ("Start Reading")
- `buttonLink`: string - The URL the button links to
- `imagePath`: string - Path to the robot image
- `layoutClass`: string - CSS class for the split layout (70% right / 30% left)

**Validation Rules**:
- Title must not be empty
- Subtitle must not be empty
- Button text must not be empty

### 2. Module Card
**Description**: Represents a single module as a card with enhanced styling

**Fields**:
- `id`: string - Unique identifier for the module
- `title`: string - The module title
- `description`: string - Brief description of the module
- `imagePath`: string - Path to the custom module image
- `link`: string - URL to the module content
- `cardStyle`: object - CSS styling properties (background, border, etc.)

**Validation Rules**:
- Title must not be empty
- Image path must be valid
- Each card must have consistent dimensions

### 3. Sidebar Category
**Description**: A category in the sidebar that can be expanded or collapsed

**Fields**:
- `id`: string - Unique identifier for the category
- `title`: string - Title of the category
- `items`: array - List of sub-modules in the category
- `collapsed`: boolean - Whether the category is currently collapsed
- `defaultExpanded`: boolean - Whether the category should be expanded by default

**Validation Rules**:
- Title must not be empty
- Items must contain at least one element
- Only "Introduction" should have defaultExpanded set to true

### 4. Module Item
**Description**: An individual item within a sidebar category

**Fields**:
- `id`: string - Unique identifier
- `title`: string - Title of the module item
- `path`: string - URL path to the module
- `sidebarId`: string - The parent sidebar category ID

**Validation Rules**:
- Title must not be empty
- Path must be a valid URL path
- Sidebar ID must reference a valid category

## State Transitions

### Sidebar Category
- **Expanded** → **Collapsed**: When user clicks the category header
- **Collapsed** → **Expanded**: When user clicks the category header
- **Default**: All categories are collapsed on initial page load, except for "Introduction" which is expanded

## Relationships

```
Sidebar Category (1) → (0..n) Module Item
Module Item (1) → (1) Module Card
Hero Section Content (1) → (0..1) Module Card (optional reference for "Start Reading" button)
```

## UI State Management

### Hero Section State
- `isLoading`: boolean - Whether the hero section content is loading
- `imageLoaded`: boolean - Whether the custom robot image has loaded

### Module Cards State
- `hoveredCardId`: string | null - The ID of the card currently being hovered over
- `allImagesLoaded`: boolean - Whether all custom images have loaded

### Sidebar State
- `expandedCategories`: array - List of category IDs that are currently expanded
- `activeModule`: string - The currently selected module path