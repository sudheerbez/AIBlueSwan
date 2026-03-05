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
            <div className="glass-card progress-feed">
                <h3 style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12 }}>
                    Live Progress
                </h3>
                <div className="empty-state" style={{ padding: '24px 0' }}>
                    <div className="empty-icon" style={{ fontSize: '2rem' }}>🔄</div>
                    <p style={{ fontSize: '0.78rem' }}>Events will appear here in real-time</p>
                </div>
            </div>
        );
    }

    return (
        <div className="glass-card progress-feed" ref={feedRef}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12 }}>
                Live Progress ({events.length} events)
            </h3>
            {events.map((event, i) => {
                const agent = event.agent || (event.type === 'pipeline_started' ? 'System' : event.type === 'pipeline_finished' ? 'System' : 'System');
                const dotClass = agentColors[agent] || 'system';
                return (
                    <div className="feed-item" key={i}>
                        <div className={`feed-dot ${dotClass}`} />
                        <div className="feed-content">
                            <div className="feed-agent">{agent.replace('Agent', '')}</div>
                            <div className="feed-message">{eventToMessage(event)}</div>
                        </div>
                        <div className="feed-time">{formatTime(event.timestamp)}</div>
                    </div>
                );
            })}
        </div>
    );
}
