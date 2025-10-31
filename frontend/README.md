# LLM TTS Service - React Frontend

Modern React frontend for visualizing Tree-of-Thoughts and other reasoning strategies.

## Features

- 🌳 **Interactive ToT Visualization** - ReactFlow-based tree with smooth interactions
- 🔍 **Node Details** - Click any node to see full state with markdown rendering
- 🎨 **Color Coding** - Visual feedback for scores and selection status
- 🔄 **Markdown Toggle** - Switch between plain text and rendered markdown
- 📍 **MiniMap & Controls** - Built-in navigation tools

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

The app will be available at http://localhost:5173/

## Components

- `App.jsx` - Main app with header and reasoning strategy selector
- `TotTreeVisualization.jsx` - ReactFlow tree visualization
- `NodeCard.jsx` - Modal for displaying node details with markdown support

## Data Format

The app expects ToT data in this format:

```javascript
{
  question: "Question text",
  nodes: [
    {
      id: 0,
      step: 0,
      score: 0.0,
      state: "**Bold text** in markdown",
      is_root: true,
      is_selected: true,
      is_final: false
    },
    // ...
  ],
  edges: [
    { from: 0, to: 1 },
    // ...
  ],
  config: {
    beam_width: 3,
    steps: 3,
    method_generate: "propose"
  }
}
```

## Future Plans

- Connect to backend API for live reasoning
- Support other strategies (CoT, Self-Consistency, DeepConf)
- Export visualizations
- Real-time streaming visualization
- Comparison views for multiple strategies

## Tech Stack

- React 18
- Vite
- ReactFlow
- React Markdown
