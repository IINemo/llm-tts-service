import { useState, useCallback, useMemo, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './TotTreeVisualization.css';
import NodeCard from './NodeCard';

const TotTreeVisualization = ({ data }) => {
  const [selectedNode, setSelectedNode] = useState(null);
  const [currentTimestamp, setCurrentTimestamp] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [maxTimestamp, setMaxTimestamp] = useState(0);

  // Calculate max timestamp on data change
  useEffect(() => {
    if (data && data.nodes && data.nodes.length > 0) {
      const timestamps = data.nodes.map(n => n.timestamp ?? n.step ?? 0);
      const max = Math.max(...timestamps);
      setMaxTimestamp(max);
      setCurrentTimestamp(max); // Start at the end
    }
  }, [data]);

  // Auto-play animation
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentTimestamp(prev => {
        if (prev >= maxTimestamp) {
          setIsPlaying(false);
          return maxTimestamp;
        }
        return prev + 1;
      });
    }, 1000); // 1 second per step

    return () => clearInterval(interval);
  }, [isPlaying, maxTimestamp]);

  // Convert ToT data to ReactFlow format
  const { nodes: initialNodes, edges: initialEdges, allNodes: allNodesData } = useMemo(() => {
    if (!data || !data.nodes) {
      return { nodes: [], edges: [], allNodes: [] };
    }

    console.log('TotTreeVisualization: Processing data', {
      nodeCount: data.nodes.length,
      edgeCount: data.edges?.length || 0
    });

    // Create a map of node IDs to their selection status for edge styling
    const nodeSelectionMap = {};
    data.nodes.forEach(node => {
      nodeSelectionMap[node.id] = node.is_selected;
    });

    // Create nodes for ReactFlow
    const nodes = data.nodes.map((node) => {
      // Determine node color based on state
      let bgColor = '#e2e8f0'; // default gray
      if (node.is_root) {
        bgColor = '#93c5fd'; // light blue
      } else if (node.is_final) {
        bgColor = '#fbbf24'; // gold
      } else if (node.score > 10) {
        bgColor = '#86efac'; // green for high score
      } else if (node.score > 1) {
        bgColor = '#fde047'; // yellow for medium
      } else {
        bgColor = '#fca5a5'; // red for low score
      }

      const borderColor = node.is_selected ? '#2563eb' : '#9ca3af';
      const borderWidth = node.is_selected ? 3 : 1;

      return {
        id: `node-${node.id}`,
        type: 'default',
        data: {
          label: node.is_root ? 'ROOT' : `N${node.id}`,
          ...node,
        },
        position: { x: 0, y: 0 }, // Will be calculated by layout
        style: {
          background: bgColor,
          border: `${borderWidth}px solid ${borderColor}`,
          borderRadius: '8px',
          padding: '10px',
          minWidth: '60px',
          fontSize: '12px',
          fontWeight: node.is_selected ? 'bold' : 'normal',
          // Add shadow for selected nodes to make them stand out more
          boxShadow: node.is_selected ? '0 0 12px rgba(37, 99, 235, 0.6)' : 'none',
        },
      };
    });

    // Create edges with highlighting for selected path
    const edges = data.edges.map((edge, idx) => {
      // Check if this edge is part of the selected path
      const isSelectedPath = nodeSelectionMap[edge.from] && nodeSelectionMap[edge.to];

      return {
        id: `edge-${idx}`,
        source: `node-${edge.from}`,
        target: `node-${edge.to}`,
        type: 'smoothstep',
        animated: isSelectedPath, // Animate edges on selected path
        style: {
          stroke: isSelectedPath ? '#2563eb' : '#9ca3af', // Blue for selected path
          strokeWidth: isSelectedPath ? 4 : 2, // Thicker for selected path
        },
        markerEnd: isSelectedPath ? {
          type: 'arrowclosed',
          color: '#2563eb',
        } : undefined,
      };
    });

    // Simple tree layout algorithm
    const layoutNodes = calculateTreeLayout(nodes, edges, data.nodes);

    return { nodes: layoutNodes, edges, allNodes: data.nodes };
  }, [data]);

  // Filter nodes and edges based on current timestamp
  const { visibleNodes, visibleEdges } = useMemo(() => {
    if (!initialNodes || !initialEdges) {
      return { visibleNodes: [], visibleEdges: [] };
    }

    // Filter nodes by timestamp
    const visibleNodes = initialNodes.filter(node => {
      const nodeData = allNodesData.find(n => n.id === node.data.id);
      if (!nodeData) return false;
      const nodeTimestamp = nodeData.timestamp ?? nodeData.step ?? 0;
      return nodeTimestamp <= currentTimestamp;
    });

    // Filter edges: only show if both source and target nodes are visible
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
    const visibleEdges = initialEdges.filter(edge =>
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    );

    return { visibleNodes, visibleEdges };
  }, [initialNodes, initialEdges, currentTimestamp, allNodesData]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update displayed nodes/edges when visibility changes
  useEffect(() => {
    setNodes(visibleNodes);
    setEdges(visibleEdges);
  }, [visibleNodes, visibleEdges, setNodes, setEdges]);

  // Handle node click
  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node.data);
  }, []);

  // Time control handlers
  const handlePlayPause = () => {
    if (currentTimestamp >= maxTimestamp) {
      setCurrentTimestamp(0); // Reset to beginning
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setCurrentTimestamp(0);
    setIsPlaying(false);
  };

  const handleSliderChange = (e) => {
    setCurrentTimestamp(parseInt(e.target.value));
    setIsPlaying(false);
  };

  return (
    <div className="tot-tree-container">
      {/* Time Control Panel */}
      <div className="time-control-panel">
        <div className="time-controls">
          <button
            className="control-button"
            onClick={handleReset}
            disabled={currentTimestamp === 0}
            title="Reset to beginning"
          >
            ⏮
          </button>
          <button
            className="control-button play-pause"
            onClick={handlePlayPause}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
          <div className="timestamp-display">
            Step {currentTimestamp} / {maxTimestamp}
          </div>
        </div>
        <div className="time-slider-container">
          <input
            type="range"
            min="0"
            max={maxTimestamp}
            value={currentTimestamp}
            onChange={handleSliderChange}
            className="time-slider"
            disabled={maxTimestamp === 0}
          />
          <div className="slider-labels">
            <span>Start</span>
            <span>Step {currentTimestamp}</span>
            <span>End</span>
          </div>
        </div>
      </div>

      {/* Tree Visualization */}
      <div className="tree-visualization">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          fitView
          minZoom={0.1}
          maxZoom={2}
        >
          <Background />
          <Controls />
          <MiniMap nodeColor={(node) => node.style.background} />
        </ReactFlow>
      </div>

      {selectedNode && (
        <NodeCard
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
};

// Hierarchical tree layout based on parent-child relationships
function calculateTreeLayout(nodes, edges, originalNodes) {
  const HORIZONTAL_SPACING = 200;
  const VERTICAL_SPACING = 120;

  console.log('[TreeLayout] Input:', {
    nodeCount: nodes.length,
    edgeCount: edges.length,
    sampleNode: nodes[0],
    sampleEdge: edges[0]
  });

  // Build parent-child map from edges
  const childrenMap = {};
  edges.forEach((edge) => {
    const parentId = edge.source;
    if (!childrenMap[parentId]) {
      childrenMap[parentId] = [];
    }
    childrenMap[parentId].push(edge.target);
  });

  console.log('[TreeLayout] ChildrenMap:', childrenMap);

  // Find root node(s)
  const allChildren = new Set(Object.values(childrenMap).flat());
  const roots = nodes.filter((n) => !allChildren.has(n.id));

  console.log('[TreeLayout] Roots found:', roots.map(r => r.id));

  // Assign horizontal positions using recursive tree layout
  const positions = {};
  let nextXOffset = 0;

  function layoutSubtree(nodeId, depth) {
    const children = childrenMap[nodeId] || [];

    if (children.length === 0) {
      // Leaf node: assign next available x position
      const x = nextXOffset * HORIZONTAL_SPACING;
      nextXOffset++;
      positions[nodeId] = { x, y: depth * VERTICAL_SPACING };
      return x;
    }

    // Layout children first
    const childXPositions = children.map((childId) =>
      layoutSubtree(childId, depth + 1)
    );

    // Position this node centered over its children
    const minChildX = Math.min(...childXPositions);
    const maxChildX = Math.max(...childXPositions);
    const x = (minChildX + maxChildX) / 2;

    positions[nodeId] = { x, y: depth * VERTICAL_SPACING };
    return x;
  }

  // Layout from each root
  roots.forEach((root) => layoutSubtree(root.id, 0));

  // Apply positions to nodes
  const positioned = nodes.map((node) => ({
    ...node,
    position: positions[node.id] || { x: 0, y: 0 },
  }));

  return positioned;
}

export default TotTreeVisualization;
