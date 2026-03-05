import {
    ResponsiveContainer,
    AreaChart,
    Area,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: 'rgba(13, 13, 24, 0.95)',
            border: '1px solid rgba(124, 58, 237, 0.3)',
            borderRadius: 8,
            padding: '10px 14px',
            fontSize: '0.78rem',
            fontFamily: "'JetBrains Mono', monospace",
        }}>
            <div style={{ color: '#94a3b8', marginBottom: 4 }}>Day {label}</div>
            <div style={{ color: '#a78bfa', fontWeight: 700 }}>
                ${payload[0].value?.toLocaleString()}
            </div>
        </div>
    );
};

export default function EquityChart({ data }) {
    if (!data || data.length === 0) {
        return (
            <div className="glass-card chart-container">
                <h3>Equity Curve</h3>
                <div className="empty-state" style={{ padding: '32px 0' }}>
                    <div className="empty-icon" style={{ fontSize: '2rem' }}>📈</div>
                    <p style={{ fontSize: '0.78rem' }}>Run a backtest to see the equity curve</p>
                </div>
            </div>
        );
    }

    return (
        <div className="glass-card chart-container">
            <h3>Equity Curve</h3>
            <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#7c3aed" stopOpacity={0.35} />
                            <stop offset="95%" stopColor="#7c3aed" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgba(255,255,255,0.04)"
                        vertical={false}
                    />
                    <XAxis
                        dataKey="day"
                        tick={{ fill: '#475569', fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                    />
                    <YAxis
                        tick={{ fill: '#475569', fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                        width={60}
                        tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                        type="monotone"
                        dataKey="equity"
                        stroke="#7c3aed"
                        strokeWidth={2}
                        fill="url(#equityGrad)"
                        dot={false}
                        animationDuration={800}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}
