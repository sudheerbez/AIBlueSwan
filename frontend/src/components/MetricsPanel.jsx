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
        <>
            {cards.map(card => (
                <div key={card.label} className="rounded-xl border border-border bg-card text-card-foreground shadow-sm">
                    <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
                        <h3 className="tracking-tight text-sm font-medium">{card.label}</h3>
                    </div>
                    <div className="p-6 pt-0">
                        <div className={`text-2xl font-bold ${card.color === 'positive' ? 'text-green-600 dark:text-green-500' :
                                card.color === 'negative' ? 'text-red-600 dark:text-red-500' :
                                    'text-foreground'
                            }`}>
                            {card.value}
                        </div>
                    </div>
                </div>
            ))}
        </>
    );
}
