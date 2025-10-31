import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './NodeCard.css';

const NodeCard = ({ node, onClose }) => {
  const [renderMode, setRenderMode] = useState('plain');

  const toggleRenderMode = () => {
    setRenderMode(renderMode === 'plain' ? 'rendered' : 'plain');
  };

  const getStatusText = () => {
    const statuses = [];
    if (node.is_root) statuses.push('ROOT');
    if (node.is_final) statuses.push('FINAL');
    if (node.is_selected) statuses.push('SELECTED');
    return statuses.length > 0 ? statuses.join(', ') : 'Generated';
  };

  return (
    <div className="node-card-overlay" onClick={onClose}>
      <div className="node-card" onClick={(e) => e.stopPropagation()}>
        <div className="node-card-header">
          <h3>Node {node.id}</h3>
          <div className="node-card-actions">
            <button
              className="toggle-render-btn"
              onClick={toggleRenderMode}
              title={renderMode === 'plain' ? 'Render Markdown' : 'Show Plain Text'}
            >
              {renderMode === 'plain' ? 'RENDER' : 'PLAIN'}
            </button>
            <button className="close-btn" onClick={onClose} title="Close">
              ×
            </button>
          </div>
        </div>

        <div className="node-card-body">
          <div className="info-row">
            <span className="info-label">Step:</span>
            <span>{node.step}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Score:</span>
            <span>{node.score.toFixed(3)}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Status:</span>
            <span>{getStatusText()}</span>
          </div>

          <div className="state-section">
            <div className="info-label">State (click to select & copy):</div>
            {renderMode === 'plain' ? (
              <div className="state-content plain">
                {node.state || <em>(empty)</em>}
              </div>
            ) : (
              <div className="state-content rendered">
                <ReactMarkdown>{node.state || '**(empty)**'}</ReactMarkdown>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NodeCard;
