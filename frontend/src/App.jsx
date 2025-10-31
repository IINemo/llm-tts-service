import { useState } from 'react'
import './App.css'
import TotTreeVisualization from './components/TotTreeVisualization'
import mockTotData from './data/mockTotData'

function App() {
  const [totData, setTotData] = useState(mockTotData)

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌳 Tree-of-Thoughts Visualization</h1>
        <div className="controls">
          <select className="reasoning-select">
            <option value="tot">Tree-of-Thoughts</option>
            <option value="cot">Chain-of-Thought</option>
            <option value="self-consistency">Self-Consistency</option>
            <option value="deepconf">DeepConf</option>
          </select>
        </div>
      </header>

      <main className="app-main">
        <TotTreeVisualization data={totData} />
      </main>
    </div>
  )
}

export default App
