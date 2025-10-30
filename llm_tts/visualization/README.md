# Reasoning Strategy Visualization

This module provides interactive visualization tools for reasoning strategies implemented in the LLM TTS service.

## Tree-of-Thoughts Visualizer

The `TotVisualizer` creates an interactive Plotly-based visualization of the Tree-of-Thoughts beam search process, showing:

- **Nodes**: States at each reasoning step with their evaluation scores
- **Edges**: Parent-child relationships showing how states evolve
- **Colors**: Score-based color coding (green=high score, red=low score)
- **Interactivity**: Hover tooltips, zoom, pan to explore the reasoning tree
- **Metadata**: Step number, score, selection status for each node

### Usage

```python
from llm_tts.strategies.tree_of_thoughts.strategy import StrategyTreeOfThoughts
from llm_tts.visualization import TotVisualizer

# Run ToT strategy
strategy = StrategyTreeOfThoughts(...)
result = strategy.generate_trajectory(prompt)

# Create visualizer
visualizer = TotVisualizer(
    width=1400,           # Figure width in pixels
    height=900,           # Figure height in pixels
    show_state_preview=True,  # Show state preview in node labels
    max_state_chars=60,   # Max characters in preview
)

# Generate visualization
fig = visualizer.visualize(
    result,
    output_path="tot_tree.html",  # Save to HTML file
    title="My ToT Reasoning",      # Custom title
    show=False,                     # Don't auto-open browser
)
```

### Quick Demo

A demo script is provided for quick testing:

```bash
python scripts/demo_tot_viz.py
```

This runs a minimal ToT search (2 steps, beam width 2) and generates an HTML visualization.

### Visualization Features

1. **Tree Structure**: Shows the complete beam search exploration
   - Root node at the top (initial empty state)
   - Each level represents a reasoning step
   - Branches show alternative reasoning paths explored

2. **Color Coding**:
   - Green: High-scoring states (promising paths)
   - Yellow: Medium-scoring states
   - Red: Low-scoring states (less promising paths)
   - Light blue: Root node
   - Gold: Final states (leaf nodes)

3. **Node Information** (visible on hover):
   - Node ID and step number
   - Evaluation score
   - Selection status (SELECTED/FINAL/Generated)
   - Full state text with reasoning steps

4. **Interactive Controls**:
   - Click and drag to pan
   - Scroll to zoom in/out
   - Double-click to reset view
   - Hover over nodes for detailed information

### Output Format

The visualizer generates a standalone HTML file that can be:
- Opened in any modern web browser
- Shared with colleagues
- Embedded in reports or presentations
- Archived for later analysis

### Integration with Scripts

Both manual test scripts include visualization:

1. **test_generic_tot.py**: Full-featured test with visualization
   ```bash
   python scripts/test_generic_tot.py "Your question here"
   # Generates: tot_visualization_Your_question_here.html
   ```

2. **demo_tot_viz.py**: Quick demo with minimal configuration
   ```bash
   python scripts/demo_tot_viz.py
   # Generates: tot_demo_visualization.html
   ```

### Requirements

- `plotly>=5.0.0` (automatically installed with the package)
- Modern web browser for viewing HTML output

### Future Enhancements

Planned visualizers for other reasoning strategies:
- Chain-of-Thought visualizer (linear path visualization)
- Self-Consistency visualizer (consensus visualization)
- DeepConf visualizer (confidence distribution plots)
