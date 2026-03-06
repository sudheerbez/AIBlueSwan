export default function CritiquePanel({ critique }) {
    if (!critique) {
        return (
            <div className="rounded-xl border border-border bg-card text-card-foreground shadow-sm p-6 flex flex-col min-h-[300px]">
                <h3 className="tracking-tight text-lg font-semibold mb-4">Analysis Critique</h3>
                <div className="flex flex-col items-center justify-center flex-1 text-muted-foreground space-y-4">
                    <div className="text-4xl">🔍</div>
                    <p className="text-sm">Critique will appear after backtest evaluation</p>
                </div>
            </div>
        );
    }

    return (
        <div className="rounded-xl border border-border bg-card text-card-foreground shadow-sm p-6 flex flex-col">
            <div className="flex items-center gap-3 mb-6">
                <h3 className="tracking-tight text-lg font-semibold">
                    Analysis Critique
                </h3>
                <span
                    className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold
                        ${critique.is_success
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                            : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                        }`}
                >
                    <span className={`h-1.5 w-1.5 rounded-full ${critique.is_success ? 'bg-green-500' : 'bg-red-500'}`} />
                    {critique.decision?.replace('_', ' ')}
                </span>
            </div>

            {/* Suggestions */}
            {critique.suggestions?.length > 0 && (
                <div className="mb-6">
                    <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
                        Suggestions
                    </div>
                    <ul className="space-y-2 text-sm text-foreground">
                        {critique.suggestions.map((s, i) => (
                            <li key={i} className="flex gap-2">
                                <span className="text-muted-foreground mt-0.5">•</span>
                                <span>{s}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Biases */}
            {critique.potential_biases?.length > 0 && (
                <div>
                    <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
                        Potential Biases
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {critique.potential_biases.map((b, i) => (
                            <span key={i} className="inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 bg-secondary text-secondary-foreground">
                                {b}
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
