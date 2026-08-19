# Frontend: Design and Styling

## File Paths: `frontend/styles.css`, `frontend/index.html`
## Status: Active / Stable

## Description
The UI uses a completely custom, framework-free design system prioritizing a premium, modern aesthetic. The layout is optimized for an uninterrupted chat experience (no manual mode switching) with a collapsable sidebar for history.

## Core CSS Principles

### 1. CSS Variables (Theming)
All colors, sizing, and shadows are mapped to CSS variables (`:root`). 
Themes are toggled by changing the `data-theme` attribute on the `<html>` element (e.g., `data-theme="light"`).
- Variables like `--bg`, `--bg-2`, `--border`, and `--accent` dynamically swap values depending on the active theme.

### 2. Layout Structure
- **CSS Grid/Flexbox**: The layout relies on flexbox for the main container (`.main`) and sidebar (`.sidebar`).
- **Responsive Breakpoints**: At `max-width: 768px`, the sidebar changes from a fixed left pane to an absolute-positioned slide-out drawer, toggled via a hamburger menu.

### 3. Glassmorphism & Animations
- The chat input (`.composer-wrap`) uses `backdrop-filter: blur(16px)` to achieve a frosted glass effect overlaid on top of chat messages.
- Subtle CSS transitions (`transition: all 0.2s ease`) are used on buttons, hovers, and input focus states to provide tactile feedback.
- Toast notifications (`.toast`) slide in from the bottom with a smooth cubic-bezier transform.

## HTML Structure (`index.html`)
1. **Sidebar (`<aside class="sidebar">`)**: Houses new chat button, history list, document upload area, settings sliders, and backend connection status.
2. **Main (`<main class="main">`)**: 
   - **Topbar**: Mobile menu toggle, Export, and Graph buttons.
   - **Welcome Screen**: Empty state showing capabilities and example prompts.
   - **Chat Area**: Where `.msg` nodes are dynamically injected by `app.js`.
   - **Composer**: Fixed to the bottom, containing the textarea and send button.
3. **Modals**: Hidden by default. Used for paper abstracts (`#paperModal`), knowledge graphs (`#kgOverlay`), and authentication (`#loginOverlay`).
