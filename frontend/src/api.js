const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

export async function startPipeline({ capital, maxIterations, universe, llmProvider, apiKey }) {
    const res = await fetch(`${API_BASE}/api/pipeline/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            capital: Number(capital),
            max_iterations: Number(maxIterations),
            universe,
            llm_provider: llmProvider,
            api_key: apiKey,
        }),
    });
    if (!res.ok) throw new Error(`Start failed: ${res.status}`);
    return res.json();
}

export async function getPipelineStatus(runId) {
    const res = await fetch(`${API_BASE}/api/pipeline/status/${runId}`);
    if (!res.ok) throw new Error(`Status failed: ${res.status}`);
    return res.json();
}

export async function getPipelineResults(runId) {
    const res = await fetch(`${API_BASE}/api/pipeline/results/${runId}`);
    if (!res.ok && res.status !== 202) throw new Error(`Results failed: ${res.status}`);
    return res.json();
}

export async function getPipelineHistory() {
    const res = await fetch(`${API_BASE}/api/pipeline/history`);
    if (!res.ok) throw new Error(`History failed: ${res.status}`);
    return res.json();
}

export function subscribeToPipeline(runId, onEvent, onClose) {
    const ws = new WebSocket(`${WS_BASE}/ws/pipeline/${runId}`);

    ws.onmessage = (e) => {
        try {
            const event = JSON.parse(e.data);
            onEvent(event);
        } catch (err) {
            console.error('WS parse error:', err);
        }
    };

    ws.onclose = () => {
        if (onClose) onClose();
    };

    ws.onerror = (err) => {
        console.error('WS error:', err);
    };

    return ws;
}
