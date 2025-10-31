import { useState, useCallback, useMemo } from 'react';
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

  // Convert ToT data to ReactFlow format
  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    if (!data || !data.nodes) return { nodes: [], edges: [] };

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
        },
      };
    });

    // Create edges
    const edges = data.edges.map((edge, idx) => ({
      id: `edge-${idx}`,
      source: `node-${edge.from}`,
      target: `node-${edge.to}`,
      type: 'smoothstep',
      animated: false,
      style: {
        stroke: '#9ca3af',
        strokeWidth: 2,
      },
    }));

    // Simple tree layout algorithm
    const layoutNodes = calculateTreeLayout(nodes, edges, data.nodes);

    return { nodes: layoutNodes, edges };
  }, [data]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Handle node click
  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node.data);
  }, []);

  return (
    <div className="tot-tree-container">
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

      {selectedNode && (
        <NodeCard
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
};

// Simple tree layout - position nodes in levels
function calculateTreeLayout(nodes, edges, originalNodes) {
  const HORIZONTAL_SPACING = 150;
  const VERTICAL_SPACING = 100;

  // Group nodes by step (level)
  const nodesByStep = {};
  originalNodes.forEach((node) => {
    if (!nodesByStep[node.step]) {
      nodesByStep[node.step] = [];
    }
    nodesByStep[node.step].push(node.id);
  });

  // Calculate positions
  const positioned = nodes.map((node) => {
    const nodeData = originalNodes.find((n) => n.id === node.data.id);
    const step = nodeData.step;
    const nodesInStep = nodesByStep[step];
    const indexInStep = nodesInStep.indexOf(nodeData.id);

    const totalWidth = (nodesInStep.length - 1) * HORIZONTAL_SPACING;
    const startX = -totalWidth / 2;

    return {
      ...node,
      position: {
        x: startX + indexInStep * HORIZONTAL_SPACING,
        y: step * VERTICAL_SPACING,
      },
    };
  });

  return positioned;
}

export default TotTreeVisualization;
