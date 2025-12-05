import { useState } from 'react'
import './App.css'
import TotTreeVisualization from './components/TotTreeVisualization'
import QueryInput from './components/QueryInput'
import { executeJobWithPolling } from './services/api'
import mockTotData from './data/mockTotData'

function App() {
  const [totData, setTotData] = useState(mockTotData)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [response, setResponse] = useState(null)
  const [progress, setProgress] = useState(null)

  const handleQuerySubmit = async (queryParams) => {
    setLoading(true)
    setError(null)
    setResponse(null)
    setProgress(null)
    setTotData(null)

    try {
      // Use polling API with progress updates
      const result = await executeJobWithPolling(
        queryParams,
        (status) => {
          // Progress callback - called on each poll
          console.log('Progress update:', status)

          // Update progress info
          setProgress({
            status: status.status,
            currentStep: status.progress.current_step,
            totalSteps: status.progress.total_steps,
            nodesExplored: status.progress.nodes_explored,
            apiCalls: status.progress.api_calls,
          })

          // Update intermediate tree if available
          if (status.intermediate_tree) {
            setTotData(status.intermediate_tree)
          }
        },
        2000 // Poll every 2 seconds
      )

      // Job completed - extract final results
      const answer = result.result.trajectory
      const reasoning_tree = result.result.reasoning_tree
      const metadata = result.result.metadata

      // Update with final tree
      if (reasoning_tree) {
        setTotData(reasoning_tree)
      }

      setResponse({
        answer,
        metadata,
      })

    } catch (err) {
      console.error('Error:', err)
      setError(err.message || 'Failed to get response from service')
    } finally {
      setLoading(false)
      setProgress(null)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌳 Tree-of-Thoughts Visualization</h1>
        <p className="subtitle">
          Visualize LLM reasoning with test-time scaling strategies
        </p>
      </header>

      <main className="app-main">
        <QueryInput onSubmit={handleQuerySubmit} loading={loading} />

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {loading && (
          <div className="loading-section">
            <div className="loading-message">
              Generating reasoning tree... This may take a minute.
            </div>
            {progress && (
              <div className="progress-info">
                <div className="progress-bar-container">
                  <div
                    className="progress-bar"
                    style={{
                      width: `${(progress.currentStep / progress.totalSteps) * 100}%`
                    }}
                  />
                </div>
                <div className="progress-stats">
                  <span>Step: {progress.currentStep}/{progress.totalSteps}</span>
                  <span>Nodes Explored: {progress.nodesExplored}</span>
                  <span>API Calls: {progress.apiCalls}</span>
                  <span className="status-badge">{progress.status}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {response && (
          <div className="response-section">
            <div className="answer-box">
              <h3>Final Answer:</h3>
              <div className="answer-content">{response.answer}</div>
            </div>

            {response.metadata && (
              <div className="metadata-box">
                <h4>Metadata:</h4>
                <ul>
                  <li><strong>Strategy:</strong> {response.metadata.strategy}</li>
                  <li><strong>Elapsed Time:</strong> {response.metadata.elapsed_time?.toFixed(2)}s</li>
                  {response.metadata.total_api_calls && (
                    <li><strong>API Calls:</strong> {response.metadata.total_api_calls}</li>
                  )}
                  {response.metadata.best_score && (
                    <li><strong>Best Score:</strong> {response.metadata.best_score?.toFixed(2)}</li>
                  )}
                </ul>
              </div>
            )}
          </div>
        )}

        {totData && (
          <div className="visualization-section">
            <h3>Reasoning Tree Visualization</h3>
            <TotTreeVisualization data={totData} />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
