export default function StrategyCard({ hypothesis }) {
    if (!hypothesis) {
        return (
            <div className="glass-card strategy-card">
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12 }}>
                    Current Strategy
                </div>
                <div className="empty-state" style={{ padding: '24px 0' }}>
                    <div className="empty-icon" style={{ fontSize: '2rem' }}>🧪</div>
                    <p style={{ fontSize: '0.78rem' }}>Awaiting hypothesis generation</p>
                </div>
            </div>
        );
    }

    return (
        <div className="glass-card strategy-card">
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
                Current Strategy
            </div>
            <div className="strategy-title">{hypothesis.title}</div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                {hypothesis.rationale}
            </p>
            <div className="factor-tags">
                {(hypothesis.factors || []).map(f => (
                    <span key={f} className="factor-tag">{f}</span>
                ))}
            </div>
            <div className="strategy-logic">{hypothesis.formula_logic}</div>
        </div>
    );
}
