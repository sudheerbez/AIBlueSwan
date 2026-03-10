import { useEffect, useRef, useState, useCallback } from 'react';
import {
  createChart,
  createSeriesMarkers,
  ColorType,
  CrosshairMode,
  LineStyle,
  CandlestickSeries,
  HistogramSeries,
  AreaSeries,
} from 'lightweight-charts';

/**
 * TradingChart — Professional candlestick chart with trade markers.
 *
 * Uses TradingView's open-source Lightweight Charts (Apache 2.0).
 * Shows OHLCV candlesticks, volume bars, entry/exit markers,
 * and an equity curve overlay.
 *
 * Props:
 *   ohlcvData  — Array of { time, open, high, low, close, volume }
 *   tradeLog   — Array of { timestamp, price, action, pnl, equity_after }
 *   equityCurve — Array of floats (equity values per bar)
 */

// ── Color palette (Robinhood-inspired) ──────────────────────────────────
const COLORS = {
  bg: '#0e1117',
  gridLines: '#1c2333',
  text: '#9ca3af',
  textStrong: '#e5e7eb',
  crosshair: '#4b5563',
  bullCandle: '#00c176',
  bearCandle: '#ff3b69',
  bullWick: '#00c176',
  bearWick: '#ff3b69',
  volume: 'rgba(76, 175, 80, 0.15)',
  equityLine: '#6366f1',
  equityArea: 'rgba(99, 102, 241, 0.08)',
  entryMarker: '#00c176',
  exitMarkerWin: '#6366f1',
  exitMarkerLoss: '#ff3b69',
  tooltipBg: '#1a1f2e',
  tooltipBorder: '#2d3548',
  pnlPositive: '#00c176',
  pnlNegative: '#ff3b69',
};

function formatCurrency(value) {
  if (value == null) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPrice(value) {
  if (value == null) return '—';
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatVolume(value) {
  if (value == null) return '—';
  if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B';
  if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M';
  if (value >= 1e3) return (value / 1e3).toFixed(1) + 'K';
  return value.toFixed(0);
}

// ── Stat pill ───────────────────────────────────────────────────────────
function StatPill({ label, value, color }) {
  return (
    <div className="flex items-center gap-1.5 rounded-md bg-[#1a1f2e] border border-[#2d3548] px-2.5 py-1">
      <span className="text-[10px] uppercase tracking-wider text-gray-500">{label}</span>
      <span className="text-xs font-semibold" style={{ color: color || '#e5e7eb' }}>{value}</span>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────
export default function TradingChart({ ohlcvData, tradeLog, equityCurve }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const [activeTab, setActiveTab] = useState('candles'); // candles | equity
  const [tradeStats, setTradeStats] = useState(null);

  // Calculate trade statistics
  useEffect(() => {
    if (!tradeLog || tradeLog.length === 0) {
      setTradeStats(null);
      return;
    }

    const exits = tradeLog.filter(t => t.action === 'exit' && t.pnl != null);
    const wins = exits.filter(t => t.pnl > 0);
    const losses = exits.filter(t => t.pnl <= 0);
    const totalPnl = exits.reduce((s, t) => s + t.pnl, 0);
    const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0;
    const avgLoss = losses.length > 0 ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0;
    const winRate = exits.length > 0 ? (wins.length / exits.length * 100) : 0;
    const profitFactor = Math.abs(avgLoss) > 0 ? (avgWin / Math.abs(avgLoss)) : avgWin > 0 ? Infinity : 0;

    setTradeStats({
      totalTrades: tradeLog.length,
      roundTrips: exits.length,
      wins: wins.length,
      losses: losses.length,
      winRate: winRate.toFixed(1),
      totalPnl,
      avgWin,
      avgLoss,
      profitFactor: profitFactor === Infinity ? '∞' : profitFactor.toFixed(2),
      bestTrade: wins.length > 0 ? Math.max(...wins.map(t => t.pnl)) : 0,
      worstTrade: losses.length > 0 ? Math.min(...losses.map(t => t.pnl)) : 0,
    });
  }, [tradeLog]);

  // Build and render the chart
  const buildChart = useCallback(() => {
    if (!chartContainerRef.current) return;

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const container = chartContainerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight || 480;

    const chart = createChart(container, {
      width,
      height,
      layout: {
        background: { type: ColorType.Solid, color: COLORS.bg },
        textColor: COLORS.text,
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: COLORS.gridLines, style: LineStyle.Dotted },
        horzLines: { color: COLORS.gridLines, style: LineStyle.Dotted },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: COLORS.crosshair,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#2d3548',
        },
        horzLine: {
          color: COLORS.crosshair,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#2d3548',
        },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.08, bottom: 0.25 },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: false,
        rightOffset: 5,
        barSpacing: 6,
        minBarSpacing: 2,
      },
      handleScroll: { vertTouchDrag: false },
    });

    chartRef.current = chart;

    if (activeTab === 'candles' && ohlcvData && ohlcvData.length > 0) {
      // ── Candlestick series (v5 API) ──
      const candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: COLORS.bullCandle,
        downColor: COLORS.bearCandle,
        borderVisible: false,
        wickUpColor: COLORS.bullWick,
        wickDownColor: COLORS.bearWick,
      });

      const candleData = ohlcvData.map(bar => ({
        time: bar.time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }));
      candleSeries.setData(candleData);

      // ── Volume histogram (v5 API) ──
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });

      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });

      const volumeData = ohlcvData.map(bar => ({
        time: bar.time,
        value: bar.volume,
        color: bar.close >= bar.open
          ? 'rgba(0, 193, 118, 0.25)'
          : 'rgba(255, 59, 105, 0.25)',
      }));
      volumeSeries.setData(volumeData);

      // ── Trade markers ──
      if (tradeLog && tradeLog.length > 0) {
        const markers = tradeLog.map(trade => {
          const isEntry = trade.action === 'entry';
          const isWin = !isEntry && trade.pnl != null && trade.pnl > 0;
          const markerColor = isEntry
            ? COLORS.entryMarker
            : isWin ? COLORS.exitMarkerWin : COLORS.exitMarkerLoss;

          const pnlText = !isEntry && trade.pnl != null
            ? ` (${trade.pnl >= 0 ? '+' : ''}${formatCurrency(trade.pnl)})`
            : '';

          return {
            time: trade.timestamp,
            position: isEntry ? 'belowBar' : 'aboveBar',
            color: markerColor,
            shape: isEntry ? 'arrowUp' : 'arrowDown',
            text: isEntry ? 'BUY' : `SELL${pnlText}`,
            size: 1.5,
          };
        });

        // Sort by time (required by lightweight-charts)
        markers.sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));
        createSeriesMarkers(candleSeries, markers);
      }

    } else if (activeTab === 'equity' && equityCurve && equityCurve.length > 0) {
      // ── Equity curve as area chart (v5 API) ──
      const areaSeries = chart.addSeries(AreaSeries, {
        topColor: 'rgba(99, 102, 241, 0.4)',
        bottomColor: 'rgba(99, 102, 241, 0.0)',
        lineColor: COLORS.equityLine,
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
        priceFormat: {
          type: 'custom',
          formatter: (price) => formatCurrency(price),
        },
      });

      // Map equity curve to time-series using OHLCV dates if available
      // equityCurve may be [{day,equity}] objects (from App.jsx) or raw floats
      const extractValue = (val) =>
        typeof val === 'number' ? val : (val?.equity ?? val?.value ?? 0);

      let equityData;
      if (ohlcvData && ohlcvData.length >= equityCurve.length) {
        equityData = equityCurve.map((val, i) => ({
          time: ohlcvData[i].time,
          value: extractValue(val),
        }));
      } else {
        // Fallback: generate sequential dates
        const startDate = new Date('2018-01-02');
        equityData = equityCurve.map((val, i) => {
          const d = new Date(startDate);
          d.setDate(d.getDate() + i);
          return {
            time: d.toISOString().slice(0, 10),
            value: extractValue(val),
          };
        });
      }
      areaSeries.setData(equityData);
    }

    // Auto-fit content
    chart.timeScale().fitContent();

    // Responsive resize
    const resizeObserver = new ResizeObserver(entries => {
      if (entries.length === 0 || !chartRef.current) return;
      const { width: newWidth, height: newHeight } = entries[0].contentRect;
      chartRef.current.applyOptions({ width: newWidth, height: newHeight });
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [ohlcvData, tradeLog, equityCurve, activeTab]);

  useEffect(() => {
    const cleanup = buildChart();
    return cleanup;
  }, [buildChart]);

  // ── Zoom handlers ──
  const handleZoomIn = useCallback(() => {
    if (!chartRef.current) return;
    const ts = chartRef.current.timeScale();
    const range = ts.getVisibleLogicalRange();
    if (!range) return;
    const span = range.to - range.from;
    const center = (range.from + range.to) / 2;
    const newSpan = Math.max(span * 0.6, 10);
    ts.setVisibleLogicalRange({ from: center - newSpan / 2, to: center + newSpan / 2 });
  }, []);

  const handleZoomOut = useCallback(() => {
    if (!chartRef.current) return;
    const ts = chartRef.current.timeScale();
    const range = ts.getVisibleLogicalRange();
    if (!range) return;
    const span = range.to - range.from;
    const center = (range.from + range.to) / 2;
    const newSpan = span * 1.6;
    ts.setVisibleLogicalRange({ from: center - newSpan / 2, to: center + newSpan / 2 });
  }, []);

  const handleResetZoom = useCallback(() => {
    if (!chartRef.current) return;
    chartRef.current.timeScale().fitContent();
  }, []);

  // ── Empty state ──
  const hasData = (ohlcvData && ohlcvData.length > 0) || (equityCurve && equityCurve.length > 0);
  if (!hasData) {
    return (
      <div className="rounded-xl border border-[#2d3548] bg-[#0e1117] text-gray-300 shadow-lg p-6 flex flex-col min-h-[520px]">
        <h3 className="tracking-tight text-lg font-semibold mb-4 text-white">Trading Chart</h3>
        <div className="flex flex-col items-center justify-center flex-1 text-gray-500 space-y-4">
          <div className="text-5xl">📊</div>
          <p className="text-sm">Run a backtest to see the professional trading chart</p>
          <p className="text-xs text-gray-600">Candlestick chart with entry/exit markers powered by TradingView</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[#2d3548] bg-[#0e1117] text-gray-300 shadow-lg flex flex-col overflow-hidden">
      {/* ── Header with tabs and stats ── */}
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold text-white tracking-tight">Trading Chart</h3>
          <div className="flex bg-[#1a1f2e] rounded-lg p-0.5 border border-[#2d3548]">
            <button
              onClick={() => setActiveTab('candles')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                activeTab === 'candles'
                  ? 'bg-[#2d3548] text-white shadow-sm'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              Candlestick
            </button>
            <button
              onClick={() => setActiveTab('equity')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                activeTab === 'equity'
                  ? 'bg-[#2d3548] text-white shadow-sm'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              Equity Curve
            </button>
          </div>
        </div>

        {/* Trade count pill */}
        {tradeStats && (
          <div className="flex items-center gap-2">
            <StatPill
              label="Win Rate"
              value={`${tradeStats.winRate}%`}
              color={parseFloat(tradeStats.winRate) >= 50 ? COLORS.pnlPositive : COLORS.pnlNegative}
            />
            <StatPill
              label="Trades"
              value={tradeStats.roundTrips}
            />
            <StatPill
              label="P/L"
              value={formatCurrency(tradeStats.totalPnl)}
              color={tradeStats.totalPnl >= 0 ? COLORS.pnlPositive : COLORS.pnlNegative}
            />
          </div>
        )}

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomIn}
            className="flex items-center justify-center w-7 h-7 rounded-md bg-[#1a1f2e] border border-[#2d3548] text-gray-400 hover:text-white hover:bg-[#2d3548] transition-all text-sm font-bold"
            title="Zoom In"
          >
            +
          </button>
          <button
            onClick={handleZoomOut}
            className="flex items-center justify-center w-7 h-7 rounded-md bg-[#1a1f2e] border border-[#2d3548] text-gray-400 hover:text-white hover:bg-[#2d3548] transition-all text-sm font-bold"
            title="Zoom Out"
          >
            &minus;
          </button>
          <button
            onClick={handleResetZoom}
            className="flex items-center justify-center h-7 px-2 rounded-md bg-[#1a1f2e] border border-[#2d3548] text-gray-400 hover:text-white hover:bg-[#2d3548] transition-all text-[10px] font-medium"
            title="Fit to Screen"
          >
            FIT
          </button>
        </div>
      </div>

      {/* ── Chart container ── */}
      <div
        ref={chartContainerRef}
        className="flex-1 min-h-[420px] w-full"
      />

      {/* ── Trade log summary bar ── */}
      {tradeStats && (
        <div className="flex items-center gap-3 px-4 py-3 border-t border-[#1c2333] overflow-x-auto">
          <StatPill label="Wins" value={tradeStats.wins} color={COLORS.pnlPositive} />
          <StatPill label="Losses" value={tradeStats.losses} color={COLORS.pnlNegative} />
          <StatPill label="Avg Win" value={formatCurrency(tradeStats.avgWin)} color={COLORS.pnlPositive} />
          <StatPill label="Avg Loss" value={formatCurrency(tradeStats.avgLoss)} color={COLORS.pnlNegative} />
          <StatPill label="Profit Factor" value={tradeStats.profitFactor} />
          <StatPill label="Best" value={formatCurrency(tradeStats.bestTrade)} color={COLORS.pnlPositive} />
          <StatPill label="Worst" value={formatCurrency(tradeStats.worstTrade)} color={COLORS.pnlNegative} />
        </div>
      )}
    </div>
  );
}
