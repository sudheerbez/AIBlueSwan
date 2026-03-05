export default function CritiquePanel({ critique }) {
    if (!critique) {
        return (
            <div className="glass-card critique-section">
                <h3>Analysis Critique</h3>
                <div className="empty-state" style={{ padding: '24px 0' }}>
                    <div className="empty-icon" style={{ fontSize: '2rem' }}>🔍</div>
                    <p style={{ fontSize: '0.78rem' }}>Critique will appear after backtest evaluation</p>
                </div>
            </div>
        );
    }

    return (
        <div className="glass-card critique-section">
            <h3>
                Analysis Critique
                <span
                    className={`status-badge ${critique.is_success ? 'success' : 'failed'}`}
                    style={{ marginLeft: 10, verticalAlign: 'middle' }}
                >
                    <span className="status-dot" />
                    {critique.decision?.replace('_', ' ')}
                </span>
            </h3>

            {/* Suggestions */}
            {critique.suggestions?.length > 0 && (
                <>
                    <div style={{
                        fontSize: '0.72rem',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        color: 'var(--text-muted)',
                        marginTop: 12,
                        marginBottom: 6,
                        letterSpacing: '0.06em',
                    }}>
                        Suggestions
                    </div>
                    <ul className="critique-list">
                        {critique.suggestions.map((s, i) => (
                            <li key={i}>{s}</li>
                        ))}
                    </ul>
                </>
            )}

            {/* Biases */}
            {critique.potential_biases?.length > 0 && (
                <div style={{ marginTop: 12 }}>
                    <div style={{
                        fontSize: '0.72rem',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        color: 'var(--text-muted)',
                        marginBottom: 6,
                        letterSpacing: '0.06em',
                    }}>
                        Potential Biases
                    </div>
                    <div>
                        {critique.potential_biases.map((b, i) => (
                            <span key={i} className="bias-tag">{b}</span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
