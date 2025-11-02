import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './NodeCard.css';

const NodeCard = ({ node, onClose }) => {
  const [renderMode, setRenderMode] = useState('rendered'); // Default to rendered mode

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

  // Parse state into thoughts with metadata
  const parseStateAsThoughts = (state) => {
    if (!state) return [{ type: 'text', content: '(empty)' }];

    const lines = state.split('\n');
    let thoughtCounter = 1;
    const thoughts = [];

    lines.forEach((line) => {
      if (line.trim().startsWith('**')) {
        // Extract content: **Some text**: Description
        const withoutStart = line.trim().substring(2);
        const endBoldIndex = withoutStart.indexOf('**');

        if (endBoldIndex !== -1) {
          const boldText = withoutStart.substring(0, endBoldIndex);
          const restOfLine = withoutStart.substring(endBoldIndex + 2);

          thoughts.push({
            type: 'thought',
            number: thoughtCounter,
            title: boldText,
            content: restOfLine,
          });
          thoughtCounter++;
        }
      } else if (line.trim()) {
        thoughts.push({ type: 'text', content: line });
      }
    });

    return thoughts;
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
                {parseStateAsThoughts(node.state).map((item, idx) => {
                  if (item.type === 'thought') {
                    return (
                      <p key={idx}>
                        <span className="thought-label">Thought {item.number}:</span>{' '}
                        <ReactMarkdown components={{ p: 'span' }}>
                          {`**${item.title}**${item.content}`}
                        </ReactMarkdown>
                      </p>
                    );
                  } else {
                    return <p key={idx}>{item.content}</p>;
                  }
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NodeCard;
