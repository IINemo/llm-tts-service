"""
Tree-of-Thoughts Visualizer using Plotly.

This module provides interactive visualization of ToT beam search,
showing the exploration tree with states, scores, and transitions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

log = logging.getLogger(__name__)


class TotVisualizer:
    """
    Interactive visualizer for Tree-of-Thoughts reasoning.

    Creates a tree/graph visualization showing:
    - Nodes: States at each step with their scores
    - Edges: Parent-child relationships
    - Colors: Score-based coloring (green=high, red=low)
    - Hover: Full state content and metadata
    """

    def __init__(
        self,
        width: int = 1400,
        height: int = 900,
        node_size: int = 20,
        show_state_preview: bool = True,
        max_state_chars: int = 10,
    ):
        """
        Initialize the visualizer.

        Args:
            width: Figure width in pixels
            height: Figure height in pixels
            node_size: Size of nodes in the visualization
            show_state_preview: Show state preview in node labels
            max_state_chars: Maximum characters to show in state preview
        """
        self.width = width
        self.height = height
        self.node_size = node_size
        self.show_state_preview = show_state_preview
        self.max_state_chars = max_state_chars

    def visualize(
        self,
        result: Dict[str, Any],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = True,
    ) -> go.Figure:
        """
        Create interactive visualization from ToT result.

        Args:
            result: Result dictionary from StrategyTreeOfThoughts.generate_trajectory()
                   Must contain 'metadata' with 'all_steps' field
            output_path: Optional path to save HTML file
            title: Optional custom title for the plot
            show: Whether to display the plot in browser

        Returns:
            Plotly Figure object

        Example:
            >>> strategy = StrategyTreeOfThoughts(...)
            >>> result = strategy.generate_trajectory(prompt)
            >>> visualizer = TotVisualizer()
            >>> fig = visualizer.visualize(result, output_path="tot_tree.html")
        """
        # Extract metadata
        metadata = result.get("metadata", {})
        generation_details = metadata.get("generation_details", {})
        all_steps = generation_details.get("all_steps", [])

        if not all_steps:
            raise ValueError(
                "No step information found in result metadata. "
                "Ensure the ToT strategy captured 'all_steps'. "
                f"Available metadata keys: {list(metadata.keys())}"
            )

        log.info(f"Visualizing ToT tree with {len(all_steps)} steps")

        # Build graph structure
        nodes, edges = self._build_graph(all_steps)

        log.info(f"Built graph with {len(nodes)} nodes and {len(edges)} edges")

        # Compute layout
        node_positions = self._compute_layout(nodes, edges, all_steps)

        # Create figure
        fig = self._create_plotly_figure(nodes, edges, node_positions, result, title)

        # Save if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save with custom HTML that makes nodes draggable
            html_content = self._generate_draggable_html(
                fig, title or "ToT Visualization"
            )
            output_file.write_text(html_content)
            log.info(f"Saved visualization to {output_file}")

        # Show if requested
        if show:
            fig.show()

        return fig

    def _build_graph(
        self, all_steps: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Tuple[int, int]]]:
        """
        Build graph structure from step information.

        Returns:
            nodes: List of node dictionaries with metadata
            edges: List of (parent_id, child_id) tuples
        """
        nodes = []
        edges = []
        node_id_counter = 0

        # Root node (empty state)
        root_node = {
            "id": node_id_counter,
            "state": "",
            "score": 0.0,
            "step": 0,
            "is_selected": True,
            "is_root": True,
            "is_final": False,
        }
        nodes.append(root_node)
        node_id_counter += 1

        # Map state content to node ID for parent tracking
        state_to_node_id = {"": 0}

        # Track selected states at each step for parent matching
        prev_selected_states = [""]

        # Process each step
        for step_idx, step_info in enumerate(all_steps):
            candidates = step_info["candidates"]
            scores = step_info["scores"]
            selected_states = step_info["selected_states"]

            # Create nodes for all candidates
            current_step_node_ids = {}

            for candidate, score in zip(candidates, scores):
                # Find parent state
                parent_state = self._find_parent_state(candidate, prev_selected_states)
                parent_id = state_to_node_id.get(parent_state, 0)

                # Check if this candidate is selected
                is_selected = candidate in selected_states

                # Create node
                node = {
                    "id": node_id_counter,
                    "state": candidate,
                    "score": score,
                    "step": step_idx + 1,
                    "is_selected": is_selected,
                    "is_root": False,
                    "is_final": False,
                }
                nodes.append(node)

                # Track mapping
                current_step_node_ids[candidate] = node_id_counter
                if is_selected:
                    state_to_node_id[candidate] = node_id_counter

                # Add edge
                edges.append((parent_id, node_id_counter))

                node_id_counter += 1

            # Update previous selected states for next iteration
            prev_selected_states = selected_states

        # Mark final nodes
        final_states_set = set(prev_selected_states)
        for node in nodes:
            if node["state"] in final_states_set and not node["is_root"]:
                node["is_final"] = True

        return nodes, edges

    def _find_parent_state(self, candidate: str, parent_states: List[str]) -> str:
        """
        Find which parent state this candidate extends.

        Args:
            candidate: Candidate state text
            parent_states: List of possible parent states

        Returns:
            Parent state text (empty string if root)
        """
        # Find the longest matching parent (handles multi-line states)
        best_match = ""
        best_match_len = 0

        for parent in parent_states:
            if parent and candidate.startswith(parent):
                if len(parent) > best_match_len:
                    best_match = parent
                    best_match_len = len(parent)

        return best_match

    def _compute_layout(
        self,
        nodes: List[Dict],
        edges: List[Tuple[int, int]],
        all_steps: List[Dict],
    ) -> Dict[int, Tuple[float, float]]:
        """
        Compute tree layout positions for nodes.

        Uses a hierarchical layout where:
        - Y position = step number (top to bottom)
        - X position = horizontal spread within each level

        Args:
            nodes: List of node dictionaries
            edges: List of edges (parent_id, child_id)
            all_steps: Original step information

        Returns:
            Dictionary mapping node_id -> (x, y) position
        """
        positions = {}

        # Group nodes by step
        nodes_by_step = {}
        for node in nodes:
            step = node["step"]
            if step not in nodes_by_step:
                nodes_by_step[step] = []
            nodes_by_step[step].append(node)

        # Assign positions level by level
        max_nodes_per_level = max(len(nodes) for nodes in nodes_by_step.values())

        for step, step_nodes in sorted(nodes_by_step.items()):
            n_nodes = len(step_nodes)

            # Y position (inverted so root is at top)
            y = -step

            # X positions (spread evenly)
            if n_nodes == 1:
                x_positions = [0.0]
            else:
                # Spread based on number of nodes at this level
                spread = min(n_nodes * 2, max_nodes_per_level * 1.5)
                x_positions = np.linspace(-spread / 2, spread / 2, n_nodes)

            # Sort nodes by parent for better visual grouping
            step_nodes_sorted = sorted(
                step_nodes,
                key=lambda n: self._get_parent_id(n["id"], edges),
            )

            for node, x in zip(step_nodes_sorted, x_positions):
                positions[node["id"]] = (x, y)

        return positions

    def _get_parent_id(self, node_id: int, edges: List[Tuple[int, int]]) -> int:
        """Get parent ID of a node."""
        for parent_id, child_id in edges:
            if child_id == node_id:
                return parent_id
        return -1

    def _create_plotly_figure(
        self,
        nodes: List[Dict],
        edges: List[Tuple[int, int]],
        positions: Dict[int, Tuple[float, float]],
        result: Dict[str, Any],
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive Plotly figure.

        Args:
            nodes: List of node dictionaries
            edges: List of edges
            positions: Node positions
            result: Full result dictionary
            title: Optional custom title

        Returns:
            Plotly Figure
        """
        # Extract edge positions
        edge_x = []
        edge_y = []

        for parent_id, child_id in edges:
            x0, y0 = positions[parent_id]
            x1, y1 = positions[child_id]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        node_hover = []

        # Get score range for color mapping
        scores = [n["score"] for n in nodes if not n["is_root"]]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        for node in nodes:
            x, y = positions[node["id"]]
            node_x.append(x)
            node_y.append(y)

            # Color based on score (normalized)
            if node["is_root"]:
                color = "lightblue"
                size = self.node_size * 1.5
            elif node["is_final"]:
                color = "gold"
                size = self.node_size * 1.3
            else:
                # Score-based coloring: green (high) to red (low)
                norm_score = (
                    (node["score"] - min_score) / score_range
                    if score_range > 0
                    else 0.5
                )
                # RGB interpolation: red (0) -> yellow (0.5) -> green (1)
                if norm_score < 0.5:
                    r = 255
                    g = int(255 * norm_score * 2)
                else:
                    r = int(255 * (1 - (norm_score - 0.5) * 2))
                    g = 255
                b = 0
                color = f"rgb({r},{g},{b})"
                size = (
                    self.node_size * 1.2
                    if node["is_selected"]
                    else self.node_size * 0.8
                )

            node_colors.append(color)
            node_sizes.append(size)

            # Label - compact format with node IDs
            if node["is_root"]:
                label = "ROOT"
            else:
                # Use compact node ID labels (N0, N1, etc.)
                label = f"N{node['id']}"
            node_text.append(label)

            # Hover info
            status = []
            if node["is_root"]:
                status.append("ROOT")
            if node["is_final"]:
                status.append("FINAL")
            if node["is_selected"]:
                status.append("SELECTED")

            hover_info = (
                f"<b>Node {node['id']}</b><br>"
                f"Step: {node['step']}<br>"
                f"Score: {node['score']:.3f}<br>"
                f"Status: {', '.join(status) if status else 'Generated'}<br>"
                f"<br><b>State:</b><br>{self._format_state_for_hover(node['state'])}"
            )
            node_hover.append(hover_info)

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            hovertext=node_hover,
            textposition="top center",
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        # Update layout
        metadata = result.get("metadata", {})
        config = metadata.get("config", {})

        if title is None:
            title = (
                f"Tree-of-Thoughts Visualization<br>"
                f"<sub>Beam Width: {config.get('beam_width', 'N/A')} | "
                f"Steps: {len(metadata.get('all_steps', []))} | "
                f"Method: {config.get('method_generate', 'N/A')}</sub>"
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            dragmode="pan",  # Enable pan mode (can switch to select in UI)
            margin=dict(b=20, l=20, r=20, t=80),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title="",
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title="",
            ),
            width=self.width,
            height=self.height,
            plot_bgcolor="white",
        )

        # Add modebar buttons for interactivity
        config = {
            "modeBarButtonsToAdd": [
                "drawopenpath",
                "eraseshape",
            ],
            "modeBarButtonsToRemove": [],
            "displaylogo": False,
        }

        # Store config for use when saving
        self._plot_config = config

        return fig

    def _get_state_preview(self, state: str) -> str:
        """Get short preview of state for labels."""
        if not state:
            return "(empty)"

        # Get last non-empty line
        lines = [line.strip() for line in state.strip().split("\n") if line.strip()]
        if not lines:
            return "(empty)"

        last_line = lines[-1]

        # Truncate if too long
        if len(last_line) > self.max_state_chars:
            return last_line[: self.max_state_chars - 3] + "..."
        return last_line

    def _generate_draggable_html(self, fig: go.Figure, title: str) -> str:
        """
        Generate HTML with draggable nodes functionality.

        Args:
            fig: Plotly figure
            title: Title for the HTML page

        Returns:
            HTML string with embedded JavaScript for draggable nodes
        """
        # Get the basic HTML from plotly
        import plotly.io as pio

        base_html = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d"],
            },
        )

        # Add custom JavaScript to make nodes draggable
        custom_script = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Wait for Plotly to render
    setTimeout(function() {
        var myDiv = document.querySelector('.plotly-graph-div');
        if (!myDiv) return;

        var draggedNode = null;
        var nodePositions = {};

        // Track node positions
        myDiv.on('plotly_hover', function(data) {
            if (data.points && data.points[0]) {
                var point = data.points[0];
                if (point.data.mode && point.data.mode.includes('markers')) {
                    // Store node info
                    draggedNode = {
                        curveNumber: point.curveNumber,
                        pointNumber: point.pointNumber,
                        x: point.x,
                        y: point.y
                    };
                }
            }
        });

        // Enable dragging with shift+click+drag
        var isDragging = false;
        var startX, startY;

        myDiv.addEventListener('mousedown', function(e) {
            if (e.shiftKey && draggedNode) {
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                e.preventDefault();
            }
        });

        document.addEventListener('mousemove', function(e) {
            if (isDragging && draggedNode) {
                var dx = (e.clientX - startX) / 5;  // Scale factor
                var dy = -(e.clientY - startY) / 5;  // Inverted Y axis

                var update = {
                    x: [[draggedNode.x + dx]],
                    y: [[draggedNode.y + dy]]
                };

                Plotly.restyle(myDiv, update, [draggedNode.curveNumber]);

                startX = e.clientX;
                startY = e.clientY;
                draggedNode.x += dx;
                draggedNode.y += dy;
            }
        });

        document.addEventListener('mouseup', function(e) {
            if (isDragging) {
                isDragging = false;
                draggedNode = null;
            }
        });

        // Add instruction text
        var instruction = document.createElement('div');
        instruction.style.cssText = 'position: fixed; top: 10px; right: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; font-family: Arial; font-size: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); z-index: 1000;';
        instruction.innerHTML = '<b>Controls:</b><br>• Pan: Click & drag<br>• Zoom: Scroll wheel<br>• Move node: Shift + Click & drag node<br>• Reset: Double-click';
        document.body.appendChild(instruction);

    }, 1000);
});
</script>
"""

        # Insert custom script before closing body tag
        html_with_script = base_html.replace("</body>", custom_script + "</body>")

        return html_with_script

    def _format_state_for_hover(self, state: str) -> str:
        """Format state text for hover display."""
        if not state:
            return "<i>(empty)</i>"

        # Limit to reasonable length
        max_chars = 500
        if len(state) > max_chars:
            state = state[:max_chars] + "..."

        # HTML escape and preserve line breaks
        state = state.replace("&", "&amp;")
        state = state.replace("<", "&lt;")
        state = state.replace(">", "&gt;")
        state = state.replace("\n", "<br>")

        return state
