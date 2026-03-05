export default function MetricsPanel({ metrics, status, iteration }) {
    const cards = [
        {
            label: 'Sharpe Ratio',
            value: metrics?.sharpe_ratio?.toFixed(4) ?? '—',
            color: metrics?.sharpe_ratio > 1.5 ? 'positive' : metrics?.sharpe_ratio > 0 ? 'neutral' : 'negative',
        },
        {
            label: 'Max Drawdown',
            value: metrics?.max_drawdown != null ? `${(metrics.max_drawdown * 100).toFixed(2)}%` : '—',
            color: metrics?.max_drawdown > -0.2 ? 'neutral' : 'negative',
        },
        {
            label: 'Annual Return',
            value: metrics?.annualized_return != null ? `${(metrics.annualized_return * 100).toFixed(2)}%` : '—',
            color: metrics?.annualized_return > 0 ? 'positive' : 'negative',
        },
        {
            label: 'Volatility',
            value: metrics?.volatility != null ? `${(metrics.volatility * 100).toFixed(2)}%` : '—',
            color: 'neutral',
        },
        {
            label: 'WFO Score',
            value: metrics?.wfo_score?.toFixed(4) ?? '—',
            color: metrics?.wfo_score > 1.0 ? 'positive' : metrics?.wfo_score > 0 ? 'neutral' : 'negative',
        },
        {
            label: 'Trades',
            value: metrics?.trades_count ?? '—',
            color: 'neutral',
        },
    ];

    return (
        <div className="metrics-grid">
            {cards.map(card => (
                <div key={card.label} className="glass-card metric-card">
                    <div className="metric-label">{card.label}</div>
                    <div className={`metric-value ${card.color}`}>{card.value}</div>
                </div>
            ))}
        </div>
    );
}
