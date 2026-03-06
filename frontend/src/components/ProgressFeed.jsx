import { useEffect, useRef } from 'react';

const agentColors = {
    SynthesisAgent: 'synthesis',
    ImplementationAgent: 'implementation',
    ValidationAgent: 'validation',
    AnalysisAgent: 'analysis',
};

function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    });
}

function eventToMessage(event) {
    if (event.type === 'pipeline_started') {
        return `Pipeline started · $${Number(event.config?.capital).toLocaleString()} · ${event.config?.universe}`;
    }
    if (event.type === 'pipeline_finished') {
        return `Pipeline ${event.status} · ${event.final_state?.iterations || 0} iterations`;
    }
    if (event.type === 'pipeline_error') {
        return `Error: ${event.error}`;
    }
    if (event.type === 'agent_update') {
        const parts = [];
        if (event.hypothesis) parts.push(`Proposed: "${event.hypothesis.title}"`);
        if (event.code) {
            parts.push(event.code.is_valid ? 'Code validated ✓' : `Code error: ${event.code.validation_error}`);
        }
        if (event.metrics) {
            parts.push(`Sharpe=${event.metrics.sharpe_ratio} · MDD=${(event.metrics.max_drawdown * 100).toFixed(1)}% · Trades=${event.metrics.trades_count}`);
        }
        if (event.critique) {
            parts.push(`Decision: ${event.critique.decision} · ${event.critique.is_success ? 'SUCCESS ✓' : 'Refining...'}`);
        }
        if (parts.length === 0) parts.push(`Status: ${event.status}`);
        return parts.join('\n');
    }
    return event.message || JSON.stringify(event);
}

export default function ProgressFeed({ events }) {
    const feedRef = useRef(null);

    useEffect(() => {
        if (feedRef.current) {
            feedRef.current.scrollTop = feedRef.current.scrollHeight;
        }
    }, [events]);

    if (!events || events.length === 0) {
        return (
            <div className="rounded-xl border border-border bg-card text-card-foreground shadow-sm p-6 flex flex-col min-h-[300px]">
                <h3 className="tracking-tight text-lg font-semibold mb-4">Live Progress</h3>
                <div className="flex flex-col items-center justify-center flex-1 text-muted-foreground space-y-4">
                    <div className="text-4xl">🔄</div>
                    <p className="text-sm">Events will appear here in real-time</p>
                </div>
            </div>
        );
    }

    return (
        <div className="rounded-xl border border-border bg-card text-card-foreground shadow-sm p-6 flex flex-col h-[400px]">
            <h3 className="tracking-tight text-lg font-semibold mb-6 flex-shrink-0">
                Live Progress <span className="text-muted-foreground text-sm font-normal">({events.length} events)</span>
            </h3>
            <div className="flex-1 overflow-y-auto space-y-4 pr-2" ref={feedRef}>
                {events.map((event, i) => {
                    const agent = event.agent || (event.type === 'pipeline_started' ? 'System' : event.type === 'pipeline_finished' ? 'System' : 'System');

                    // Simple dot color logic based on agent
                    const getDotColor = (agentName) => {
                        if (agentName.includes('Synthesis')) return 'bg-purple-500';
                        if (agentName.includes('Implementation')) return 'bg-blue-500';
                        if (agentName.includes('Validation')) return 'bg-amber-500';
                        if (agentName.includes('Analysis')) return 'bg-emerald-500';
                        if (event.type === 'error') return 'bg-red-500';
                        return 'bg-gray-400 dark:bg-gray-600';
                    };

                    return (
                        <div className="flex items-start gap-3 text-sm relative" key={i}>
                            <div className="mt-1.5 shrink-0">
                                <div className={`h-2.5 w-2.5 rounded-full ${getDotColor(agent)} ring-4 ring-background`} />
                            </div>
                            {/* Vertical connecting line */}
                            {i !== events.length - 1 && (
                                <div className="absolute top-4 left-[0.3125rem] bottom-[-1rem] w-px bg-border" />
                            )}
                            <div className="flex-1 space-y-1 pb-2">
                                <div className="flex items-center justify-between">
                                    <span className="font-semibold text-foreground">{agent.replace('Agent', '')}</span>
                                    <span className="text-xs text-muted-foreground tabular-nums">{formatTime(event.timestamp)}</span>
                                </div>
                                <div className="text-muted-foreground whitespace-pre-line leading-snug">
                                    {eventToMessage(event)}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
