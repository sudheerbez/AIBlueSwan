import { useState, useEffect, useRef, useCallback } from 'react';
import MetricsPanel from './components/MetricsPanel';
import EquityChart from './components/EquityChart';
import ProgressFeed from './components/ProgressFeed';
import StrategyCard from './components/StrategyCard';
import CritiquePanel from './components/CritiquePanel';
import { startPipeline, subscribeToPipeline, getPipelineHistory } from './api';

const DEFAULT_CONFIG = {
  capital: 100000,
  maxIterations: 2,
  universe: 'NASDAQ-100',
  llmProvider: 'Ollama',
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
    <div className="flex h-screen w-full bg-background text-foreground overflow-hidden font-sans">
      {/* Sidebar */}
      <aside className="w-80 border-r border-border bg-card/50 flex flex-col pt-6 pb-6 px-4 overflow-y-auto shrink-0 shadow-sm z-10">
        <div className="flex items-center gap-3 px-2 mb-8">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary text-primary-foreground font-bold text-xs shadow-md">AQ</div>
          <h1 className="text-xl font-bold tracking-tight">AI Blue Swan</h1>
        </div>
        <div className="flex flex-col space-y-6">
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider px-2">AI Blue Swan Config</h3>
            <div className="space-y-1.5 px-2">
              <label htmlFor="input-capital" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Initial Capital ($)
              </label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                type="number"
                value={config.capital}
                onChange={e => setConfig(c => ({ ...c, capital: e.target.value }))}
                disabled={status === 'running'}
                id="input-capital"
              />
            </div>
            <div className="space-y-1.5 px-2">
              <label htmlFor="input-iterations" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Max Iterations
              </label>
              <input
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                type="number"
                min="1"
                max="50"
                value={config.maxIterations}
                onChange={e => setConfig(c => ({ ...c, maxIterations: e.target.value }))}
                disabled={status === 'running'}
                id="input-iterations"
              />
            </div>
            <div className="space-y-1.5 px-2">
              <label htmlFor="select-universe" className="text-sm font-medium leading-none">
                Universe
              </label>
              <select
                className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
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
            <div className="space-y-1.5 px-2">
              <label htmlFor="select-llm-provider" className="text-sm font-medium leading-none">
                LLM Provider (Local)
              </label>
              <select
                className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                value={config.llmProvider}
                onChange={e => setConfig(c => ({ ...c, llmProvider: e.target.value }))}
                disabled={true}
                id="select-llm-provider"
              >
                <option value="Ollama">Ollama (Local Models)</option>
              </select>
              <p className="text-xs text-muted-foreground mt-1 px-1">Configure models in config.py.</p>
            </div>

            <div className="px-2 pt-2">
              {status !== 'running' ? (
                <button
                  className="inline-flex w-full items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground shadow hover:bg-primary/90 h-9 px-4 py-2"
                  onClick={handleStart}
                  id="btn-start"
                >
                  ▶ Start Pipeline
                </button>
              ) : (
                <button
                  className="inline-flex w-full items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90 h-9 px-4 py-2"
                  onClick={handleStop}
                  id="btn-stop"
                >
                  ■ Stop Pipeline
                </button>
              )}
            </div>
          </div>

          {/* History */}
          <div className="space-y-4 px-2 mt-8">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Run History</h3>
            {history.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No runs yet
              </p>
            ) : (
              <ul className="space-y-2">
                {history.slice(0, 10).map(run => (
                  <li
                    key={run.run_id}
                    className={`flex flex-col gap-1 p-3 rounded-md border text-sm transition-colors ${run.run_id === runId
                      ? 'border-primary bg-primary/5'
                      : 'border-border bg-card hover:bg-accent/50 cursor-pointer'
                      }`}
                    onClick={() => {/* could set past run */ }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs text-muted-foreground">#{run.run_id.substring(0, 8)}</span>
                      <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${run.status === 'success' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                        run.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                          'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                        }`}>
                        {run.status}
                      </span>
                    </div>
                    <div className="font-medium">
                      ${Number(run.capital).toLocaleString()}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-y-auto bg-background/95">
        {/* Header content moved to top of main */}
        <header className="sticky top-0 z-20 flex h-14 items-center gap-4 border-b bg-background/95 px-6 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="ml-auto flex items-center gap-2">
            {status === 'running' && (
              <span className="inline-flex items-center gap-2 rounded-full border border-yellow-200 bg-yellow-100/50 px-3 py-1 text-sm font-medium text-yellow-800 dark:border-yellow-900/50 dark:bg-yellow-900/20 dark:text-yellow-400">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-yellow-500"></span>
                </span>
                Running · Iteration {iteration + 1}
              </span>
            )}
            {status === 'success' && (
              <span className="inline-flex items-center gap-2 rounded-full border border-green-200 bg-green-100/50 px-3 py-1 text-sm font-medium text-green-800 dark:border-green-900/50 dark:bg-green-900/20 dark:text-green-400">
                <span className="h-2 w-2 rounded-full bg-green-500"></span>
                Complete
              </span>
            )}
            {status === 'failed' && (
              <span className="inline-flex items-center gap-2 rounded-full border border-red-200 bg-red-100/50 px-3 py-1 text-sm font-medium text-red-800 dark:border-red-900/50 dark:bg-red-900/20 dark:text-red-400">
                <span className="h-2 w-2 rounded-full bg-red-500"></span>
                Failed
              </span>
            )}
            {status === 'idle' && (
              <span className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-medium text-muted-foreground">
                <span className="h-2 w-2 rounded-full bg-muted-foreground/30"></span>
                Ready
              </span>
            )}
          </div>
        </header>

        <div className="flex-1 space-y-6 p-8">
          {status === 'idle' && !metrics ? (
            <div className="flex flex-col items-center justify-center h-full max-w-md mx-auto text-center space-y-4 text-muted-foreground">
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 text-primary text-3xl mb-4">
                🧬
              </div>
              <h3 className="text-2xl font-semibold text-foreground tracking-tight">Ready to Discover Alpha</h3>
              <p className="text-sm">
                Configure your pipeline parameters and click
                <strong className="text-foreground"> Start Pipeline</strong> in the sidebar to begin autonomous strategy research.
              </p>
            </div>
          ) : (
            <>
              {/* Metrics */}
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <MetricsPanel metrics={metrics} status={status} iteration={iteration} />
              </div>

              {/* Equity Chart + Strategy side by side */}
              <div className="grid gap-4 md:grid-cols-2">
                <EquityChart data={equityCurve} />
                <StrategyCard hypothesis={hypothesis} />
              </div>

              {/* Critique + Progress Feed */}
              <div className="grid gap-4 md:grid-cols-2">
                <CritiquePanel critique={critique} />
                <ProgressFeed events={events} />
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
