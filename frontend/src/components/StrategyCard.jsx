export default function StrategyCard({ hypothesis }) {
    if (!hypothesis) {
        return (
            <div className="rounded-xl border border-border bg-card text-card-foreground shadow-sm p-6 flex flex-col min-h-[300px]">
                <div className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-6">
                    Current Strategy
                </div>
                <div className="flex flex-col items-center justify-center flex-1 text-muted-foreground space-y-4">
                    <div className="text-4xl">🧪</div>
                    <p className="text-sm">Awaiting hypothesis generation</p>
                </div>
            </div>
        );
    }

    return (
        <div className="rounded-xl border border-border bg-card text-card-foreground shadow-sm p-6 flex flex-col gap-4">
            <div className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                Current Strategy
            </div>
            <div>
                <h4 className="text-lg font-bold text-foreground leading-tight">{hypothesis.title}</h4>
                <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
                    {hypothesis.rationale}
                </p>
            </div>

            {hypothesis.factors && hypothesis.factors.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                    {hypothesis.factors.map(f => (
                        <span key={f} className="inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 bg-secondary text-secondary-foreground hover:bg-secondary/80">
                            {f}
                        </span>
                    ))}
                </div>
            )}

            {hypothesis.formula_logic && (
                <div className="mt-4 rounded-md bg-muted px-4 py-3 font-mono text-sm text-muted-foreground border border-border overflow-x-auto">
                    {hypothesis.formula_logic}
                </div>
            )}
        </div>
    );
}
