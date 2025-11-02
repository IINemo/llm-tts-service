"""
Graph data structures for representing reasoning trees.

Used by strategies like Tree-of-Thoughts to track exploration paths,
scores, and parent-child relationships during search.
"""

from typing import Any, Dict, List, Optional


class ReasoningNode:
    """
    A node in a reasoning graph/tree.

    Represents a single state in the reasoning process with metadata
    about its evaluation, position in the tree, and relationships.
    """

    def __init__(
        self,
        node_id: int,
        state: str,
        parent: Optional["ReasoningNode"] = None,
        step: int = 0,
        score: float = 0.0,
        timestamp: int = 0,
        is_root: bool = False,
        is_selected: bool = False,
        is_final: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a reasoning node.

        Args:
            node_id: Unique identifier for this node
            state: The reasoning state/content at this node
            parent: Parent node (None for root)
            step: Step/depth in the reasoning tree (0 for root)
            score: Evaluation score for this state
            timestamp: Creation timestamp for animation
            is_root: Whether this is the root node
            is_selected: Whether this node is on the selected path
            is_final: Whether this is a terminal/final state
            metadata: Additional metadata
        """
        self.id = node_id
        self.state = state
        self.parent = parent
        self.children: List[ReasoningNode] = []
        self.step = step
        self.score = score
        self.timestamp = timestamp
        self.is_root = is_root
        self.is_selected = is_selected
        self.is_final = is_final
        self.metadata = metadata or {}

    def add_child(self, child: "ReasoningNode") -> None:
        """Add a child node to this node."""
        self.children.append(child)
        child.parent = self

    def get_path_from_root(self) -> List["ReasoningNode"]:
        """Get the path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def get_depth(self) -> int:
        """Get depth of this node (distance from root)."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "step": self.step,
            "state": self.state,
            "score": self.score,
            "is_root": self.is_root,
            "is_selected": self.is_selected,
            "is_final": self.is_final,
            "timestamp": self.timestamp,
            "parent_id": self.parent.id if self.parent else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"Node(id={self.id}, step={self.step}, score={self.score:.2f})"


class ReasoningGraph:
    """
    A graph/tree structure for tracking reasoning exploration.

    Maintains nodes, edges, and provides methods for traversal,
    serialization, and visualization.
    """

    def __init__(self, question: str = ""):
        """
        Initialize a reasoning graph.

        Args:
            question: The original question/problem being solved
        """
        self.question = question
        self.nodes: Dict[int, ReasoningNode] = {}
        self.root: Optional[ReasoningNode] = None
        self._next_node_id = 0

    def create_root(self, state: str = "", timestamp: int = 0) -> ReasoningNode:
        """
        Create and set the root node.

        Args:
            state: Initial state (usually empty)
            timestamp: Creation timestamp

        Returns:
            The created root node
        """
        root = ReasoningNode(
            node_id=self._get_next_id(),
            state=state,
            parent=None,
            step=0,
            score=0.0,
            timestamp=timestamp,
            is_root=True,
            is_selected=True,
            is_final=False,
        )
        self.add_node(root)
        self.root = root
        return root

    def add_node(self, node: ReasoningNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def create_child(
        self,
        parent: ReasoningNode,
        state: str,
        score: float = 0.0,
        timestamp: int = 0,
        is_selected: bool = False,
        is_final: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningNode:
        """
        Create a new child node.

        Args:
            parent: Parent node
            state: State content for new node
            score: Evaluation score
            timestamp: Creation timestamp
            is_selected: Whether on selected path
            is_final: Whether terminal state
            metadata: Additional metadata

        Returns:
            The created child node
        """
        child = ReasoningNode(
            node_id=self._get_next_id(),
            state=state,
            parent=parent,
            step=parent.step + 1,
            score=score,
            timestamp=timestamp,
            is_root=False,
            is_selected=is_selected,
            is_final=is_final,
            metadata=metadata,
        )
        self.add_node(child)
        parent.add_child(child)
        return child

    def get_node(self, node_id: int) -> Optional[ReasoningNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[ReasoningNode]:
        """Get all nodes in the graph."""
        return list(self.nodes.values())

    def get_edges(self) -> List[Dict[str, int]]:
        """
        Get all edges as parent-child pairs.

        Returns:
            List of {"from": parent_id, "to": child_id}
        """
        edges = []
        for node in self.nodes.values():
            if node.parent:
                edges.append({"from": node.parent.id, "to": node.id})
        return edges

    def get_selected_path(self) -> List[ReasoningNode]:
        """Get the selected path (nodes with is_selected=True)."""
        if not self.root:
            return []

        # Find a final selected node
        selected_finals = [
            n for n in self.nodes.values() if n.is_selected and n.is_final
        ]

        if selected_finals:
            # Get path from root to this final node
            return selected_finals[0].get_path_from_root()

        # Fallback: find any selected node and trace back
        selected = [n for n in self.nodes.values() if n.is_selected]
        if selected:
            # Return path to deepest selected node
            deepest = max(selected, key=lambda n: n.step)
            return deepest.get_path_from_root()

        return [self.root] if self.root else []

    def mark_selected_path(self, final_node: ReasoningNode) -> None:
        """
        Mark all nodes from root to final_node as selected.

        Args:
            final_node: The final node in the selected path
        """
        path = final_node.get_path_from_root()
        for node in path:
            node.is_selected = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to dictionary for serialization.

        Returns:
            Dict with "nodes", "edges", "question" keys
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": self.get_edges(),
            "question": self.question,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        nodes = list(self.nodes.values())
        return {
            "total_nodes": len(nodes),
            "total_edges": len(self.get_edges()),
            "max_depth": max((n.step for n in nodes), default=0),
            "selected_nodes": sum(1 for n in nodes if n.is_selected),
            "final_nodes": sum(1 for n in nodes if n.is_final),
        }

    def _get_next_id(self) -> int:
        """Get next available node ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ReasoningGraph(nodes={stats['total_nodes']}, depth={stats['max_depth']})"
        )
