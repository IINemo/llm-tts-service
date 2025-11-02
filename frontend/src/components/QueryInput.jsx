import { useState } from 'react';
import './QueryInput.css';

function QueryInput({ onSubmit, loading }) {
  const [question, setQuestion] = useState('');
  const [strategy, setStrategy] = useState('tree_of_thoughts');
  const [beamWidth, setBeamWidth] = useState(3);
  const [steps, setSteps] = useState(3);
  const [nGenerate, setNGenerate] = useState(5);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const strategyConfig = {};

    if (strategy === 'tree_of_thoughts' || strategy === 'tot') {
      strategyConfig.tot_beam_width = beamWidth;
      strategyConfig.tot_steps = steps;
      strategyConfig.tot_n_generate_sample = nGenerate;
      strategyConfig.tot_method_generate = 'propose';
    }

    onSubmit({
      message: question,
      strategy,
      strategyConfig,
    });
  };

  return (
    <div className="query-input">
      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <label htmlFor="question">Your Question:</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question here (e.g., What are the benefits of using Tree-of-Thoughts?)"
            rows={4}
            disabled={loading}
          />
        </div>

        <div className="settings-row">
          <div className="input-group">
            <label htmlFor="strategy">Strategy:</label>
            <select
              id="strategy"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              disabled={loading}
            >
              <option value="tree_of_thoughts">Tree-of-Thoughts</option>
              <option value="deepconf">DeepConf</option>
              <option value="chain_of_thought">Chain-of-Thought</option>
              <option value="self_consistency">Self-Consistency</option>
            </select>
          </div>

          {(strategy === 'tree_of_thoughts' || strategy === 'tot') && (
            <>
              <div className="input-group">
                <label htmlFor="beamWidth">Beam Width:</label>
                <input
                  type="number"
                  id="beamWidth"
                  value={beamWidth}
                  onChange={(e) => setBeamWidth(parseInt(e.target.value))}
                  min={1}
                  max={10}
                  disabled={loading}
                />
              </div>

              <div className="input-group">
                <label htmlFor="steps">Steps:</label>
                <input
                  type="number"
                  id="steps"
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value))}
                  min={1}
                  max={10}
                  disabled={loading}
                />
              </div>

              <div className="input-group">
                <label htmlFor="nGenerate">Candidates per State:</label>
                <input
                  type="number"
                  id="nGenerate"
                  value={nGenerate}
                  onChange={(e) => setNGenerate(parseInt(e.target.value))}
                  min={1}
                  max={10}
                  disabled={loading}
                />
              </div>
            </>
          )}
        </div>

        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Generating...' : 'Generate Response'}
        </button>
      </form>
    </div>
  );
}

export default QueryInput;
