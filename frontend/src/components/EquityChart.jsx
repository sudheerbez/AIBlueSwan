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
        <div className="bg-popover text-popover-foreground rounded-lg border border-border px-3 py-2 text-xs shadow-md">
            <div className="text-muted-foreground mb-1">Day {label}</div>
            <div className="font-bold text-primary">
                ${payload[0].value?.toLocaleString()}
            </div>
        </div>
    );
};

export default function EquityChart({ data }) {
    if (!data || data.length === 0) {
        return (
            <div className="rounded-xl border border-border bg-card text-card-foreground shadow-sm p-6 flex flex-col min-h-[300px]">
                <h3 className="tracking-tight text-lg font-semibold mb-4">Equity Curve</h3>
                <div className="flex flex-col items-center justify-center flex-1 text-muted-foreground space-y-4">
                    <div className="text-4xl">📈</div>
                    <p className="text-sm">Run a backtest to see the equity curve</p>
                </div>
            </div>
        );
    }

    return (
        <div className="rounded-xl border border-border bg-card text-card-foreground shadow-sm p-6 flex flex-col">
            <h3 className="tracking-tight text-lg font-semibold mb-6">Equity Curve</h3>
            <div className="flex-1 w-full min-h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="hsl(var(--border))"
                            vertical={false}
                        />
                        <XAxis
                            dataKey="day"
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                            tickMargin={10}
                        />
                        <YAxis
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                            axisLine={false}
                            tickLine={false}
                            tickMargin={10}
                            width={60}
                            tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Area
                            type="monotone"
                            dataKey="equity"
                            stroke="hsl(var(--primary))"
                            strokeWidth={2}
                            fill="url(#equityGrad)"
                            dot={false}
                            animationDuration={800}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
