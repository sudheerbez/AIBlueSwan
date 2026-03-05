import { useState, useEffect, useRef, useCallback } from 'react';
import MetricsPanel from './components/MetricsPanel';
import EquityChart from './components/EquityChart';
import ProgressFeed from './components/ProgressFeed';
import StrategyCard from './components/StrategyCard';
import CritiquePanel from './components/CritiquePanel';
import { startPipeline, subscribeToPipeline, getPipelineHistory } from './api';

const DEFAULT_CONFIG = {
  capital: 100000,
  maxIterations: 5,
  universe: 'NASDAQ-100',
};

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [runId, setRunId] = useState(null);
  const [status, setStatus] = useState('idle');        // idle | running | success | failed
  const [events, setEvents] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [equityCurve, setEquityCurve] = useState([]);
  const [hypothesis, setHypothesis] = useState(null);
  const [critique, setCritique] = useState(null);
  const [iteration, setIteration] = useState(0);
  const [history, setHistory] = useState([]);
  const wsRef = useRef(null);

  // Load history on mount
  useEffect(() => {
    getPipelineHistory()
      .then(data => setHistory(data.runs || []))
      .catch(() => { });
  }, []);

  const handleStart = useCallback(async () => {
    try {
      setStatus('running');
      setEvents([]);
      setMetrics(null);
      setEquityCurve([]);
      setHypothesis(null);
      setCritique(null);
      setIteration(0);

      const res = await startPipeline(config);
      setRunId(res.run_id);

      // Subscribe to WebSocket
      wsRef.current = subscribeToPipeline(
        res.run_id,
        (event) => {
          setEvents(prev => [...prev, event]);

          if (event.type === 'agent_update') {
            if (event.iteration !== undefined) setIteration(event.iteration);
            if (event.hypothesis) setHypothesis(event.hypothesis);
            if (event.metrics) setMetrics(event.metrics);
            if (event.equity_curve) {
              setEquityCurve(event.equity_curve.map((v, i) => ({
                day: i,
                equity: Number(v.toFixed(2)),
              })));
            }
            if (event.critique) setCritique(event.critique);
          }

          if (event.type === 'pipeline_finished') {
            setStatus(event.status === 'success' ? 'success' : 'failed');
            // Refresh history
            getPipelineHistory()
              .then(data => setHistory(data.runs || []))
              .catch(() => { });
          }
        },
        () => {
          if (status === 'running') setStatus('failed');
        }
      );
    } catch (err) {
      setStatus('failed');
      setEvents(prev => [...prev, {
        type: 'error',
        agent: 'System',
        timestamp: new Date().toISOString(),
        message: err.message,
      }]);
    }
  }, [config, status]);

  const handleStop = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus('idle');
  }, []);

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="app-header">
        <div className="logo">
          <div className="logo-icon">AQ</div>
          <h1>AutoQuant</h1>
        </div>
        <div className="header-status">
          {status === 'running' && (
            <span className="status-badge running">
              <span className="status-dot" />
              Running · Iteration {iteration + 1}
            </span>
          )}
          {status === 'success' && (
            <span className="status-badge success">
              <span className="status-dot" />
              Complete
            </span>
          )}
          {status === 'failed' && (
            <span className="status-badge failed">
              <span className="status-dot" />
              Failed
            </span>
          )}
          {status === 'idle' && (
            <span className="status-badge pending">
              Ready
            </span>
          )}
        </div>
      </header>

      {/* Sidebar */}
      <aside className="app-sidebar">
        <div className="controls-section">
          <h3>Pipeline Config</h3>
          <div className="input-group">
            <label>Initial Capital ($)</label>
            <input
              type="number"
              value={config.capital}
              onChange={e => setConfig(c => ({ ...c, capital: e.target.value }))}
              disabled={status === 'running'}
              id="input-capital"
            />
          </div>
          <div className="input-group">
            <label>Max Iterations</label>
            <input
              type="number"
              min="1"
              max="50"
              value={config.maxIterations}
              onChange={e => setConfig(c => ({ ...c, maxIterations: e.target.value }))}
              disabled={status === 'running'}
              id="input-iterations"
            />
          </div>
          <div className="input-group">
            <label>Universe</label>
            <select
              value={config.universe}
              onChange={e => setConfig(c => ({ ...c, universe: e.target.value }))}
              disabled={status === 'running'}
              id="select-universe"
            >
              <option value="NASDAQ-100">NASDAQ-100</option>
              <option value="S&P-500">S&P 500</option>
              <option value="DJIA-30">DJIA 30</option>
            </select>
          </div>

          {status !== 'running' ? (
            <button className="btn btn-primary" onClick={handleStart} id="btn-start">
              ▶ Start Pipeline
            </button>
          ) : (
            <button className="btn btn-danger" onClick={handleStop} id="btn-stop">
              ■ Stop Pipeline
            </button>
          )}
        </div>

        {/* History */}
        <div className="controls-section" style={{ marginTop: 24 }}>
          <h3>Run History</h3>
          {history.length === 0 ? (
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              No runs yet
            </p>
          ) : (
            <ul className="history-list">
              {history.slice(0, 10).map(run => (
                <li
                  key={run.run_id}
                  className={`history-item ${run.run_id === runId ? 'active' : ''}`}
                >
                  <div className="run-id">#{run.run_id}</div>
                  <div className="run-meta">
                    <span className={`status-badge ${run.status}`} style={{ marginRight: 6 }}>
                      <span className="status-dot" />
                      {run.status}
                    </span>
                    ${Number(run.capital).toLocaleString()}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="app-main">
        {status === 'idle' && !metrics ? (
          <div className="empty-state">
            <div className="empty-icon">🧬</div>
            <h3>Ready to Discover Alpha</h3>
            <p>
              Configure your pipeline parameters and click
              <strong> Start Pipeline</strong> to begin autonomous strategy research.
            </p>
          </div>
        ) : (
          <>
            {/* Metrics */}
            <MetricsPanel metrics={metrics} status={status} iteration={iteration} />

            {/* Equity Chart + Strategy side by side */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
              <EquityChart data={equityCurve} />
              <StrategyCard hypothesis={hypothesis} />
            </div>

            {/* Critique + Progress Feed */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <CritiquePanel critique={critique} />
              <ProgressFeed events={events} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}
