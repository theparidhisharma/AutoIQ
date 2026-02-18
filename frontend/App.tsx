import React, { useState, useEffect, useCallback } from 'react';
import {
  TelemetryData,
  AgentLog,
  RiskAnalysis,
  RiskBand,
  RCADataset,
  ViewState
} from './types';

/* =========================
   UI COMPONENTS
========================= */

const Navbar: React.FC<{ current: ViewState; setView: (v: ViewState) => void }> = ({ current, setView }) => {
  const links: { id: ViewState; label: string }[] = [
    { id: 'dashboard', label: 'Control Center' },
    { id: 'vehicle', label: 'Vehicle State' },
    { id: 'rca', label: 'RCA / CAPA' },
    { id: 'ueba', label: 'Agent Audit' }
  ];

  return (
    <nav className="flex gap-1 mb-8 bg-slate-900/50 p-1 rounded-lg border border-white/5 w-fit">
      {links.map(link => (
        <button
          key={link.id}
          onClick={() => setView(link.id)}
          className={`px-4 py-1.5 rounded-md text-xs font-bold transition-all ${
            current === link.id ? 'bg-sky-600 text-white shadow-lg' : 'text-white/40 hover:text-white/70'
          }`}
        >
          {link.label}
        </button>
      ))}
    </nav>
  );
};

const LogConsole: React.FC<{ logs: AgentLog[] }> = ({ logs }) => (
  <div className="glass rounded-xl p-4 h-full flex flex-col">
    <div className="flex justify-between items-center mb-4 border-b border-white/10 pb-2">
      <h3 className="font-bold text-sky-400 text-sm">Agent Activity Log</h3>
      <span className="text-[10px] text-white/30 uppercase tracking-widest font-bold">UEBA Monitored</span>
    </div>
    <div className="flex-1 overflow-y-auto space-y-2 mono text-[11px] scrollbar-hide">
      {logs.map(log => (
        <div key={log.id} className="border-l-2 pl-3 py-1 border-white/5 bg-white/5 rounded-r">
          <span className="text-white/20 text-[9px] block">{log.timestamp}</span>
          <span className="font-bold text-purple-400 mr-2">[{log.agent}]</span>
          <span className="text-white/70">{log.message}</span>
        </div>
      ))}
      {logs.length === 0 && (
        <div className="text-center text-white/10 mt-10 italic">
          Awaiting telemetry handshake…
        </div>
      )}
    </div>
  </div>
);

/* =========================
   MAIN APP
========================= */

export default function App() {
  const [view, setView] = useState<ViewState>('dashboard');

  const [telemetry, setTelemetry] = useState<TelemetryData>({
    airTemp: 298,
    processTemp: 308,
    rotationalSpeed: 1500,
    torque: 40,
    toolWear: 50
  });

  const [analysis, setAnalysis] = useState<RiskAnalysis | null>(null);
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [rca, setRca] = useState<RCADataset | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  /* =========================
     HELPERS
  ========================= */

  const updateField = (field: keyof TelemetryData, value: string) => {
    setTelemetry(prev => ({ ...prev, [field]: Number(value) || 0 }));
  };

  const addLog = useCallback((message: string) => {
    setLogs(prev => [
      {
        id: crypto.randomUUID(),
        agent: 'UEBA',
        message,
        timestamp: new Date().toLocaleTimeString(),
        level: 'info'
      },
      ...prev
    ].slice(0, 50));
  }, []);

  /* =========================
     RUN DIAGNOSTICS (BACKEND)
  ========================= */

  const runDiagnostics = async () => {
    setIsLoading(true);
    setLogs([]);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(telemetry)
      });

      const data = await res.json();

      setAnalysis({
        finalRisk: data.final_risk,
        mlRisk: data.ml_risk,
        ruleRisk: data.rule_risk,
        band: data.state,
        decisionTrace: [
          'Telemetry ingested',
          'ML inference executed',
          'Rule-based checks applied',
          `System state: ${data.state}`
        ]
      });

    } catch {
      addLog('Backend unreachable — SAFE MODE');
    } finally {
      setIsLoading(false);
    }
  };

  /* =========================
     UEBA LOG SYNC
  ========================= */

  useEffect(() => {
    if (!analysis) return;

    fetch('/api/ueba')
      .then(res => res.json())
      .then(data => {
        setLogs(
          data.map((l: any, i: number) => ({
            id: String(i),
            agent: 'UEBA',
            message: l.action,
            timestamp: l.timestamp,
            level: 'info'
          }))
        );
      })
      .catch(() => {});
  }, [analysis]);

  /* =========================
     RCA SYNC
  ========================= */

  useEffect(() => {
    if (analysis?.band === RiskBand.EMERGENCY) {
      fetch('/api/rca')
        .then(res => res.json())
        .then(setRca);
    } else {
      setRca(null);
    }
  }, [analysis]);

  /* =========================
     RENDER
  ========================= */

  return (
    <div className="min-h-screen p-6 bg-[#020617] text-white">
      <Navbar current={view} setView={setView} />

      {view === 'dashboard' && (
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-8">
            <button
              onClick={runDiagnostics}
              disabled={isLoading}
              className="mb-4 bg-sky-600 px-6 py-3 rounded-lg font-bold"
            >
              {isLoading ? 'ORCHESTRATING…' : 'SYNC & DIAGNOSE'}
            </button>

            {analysis && (
              <div className="glass p-6 rounded-xl">
                <div className="text-5xl font-black">
                  {analysis.finalRisk.toFixed(1)}%
                </div>
                <div className="text-sm opacity-60">
                  {analysis.band} STATE
                </div>
              </div>
            )}
          </div>

          <div className="col-span-4 h-[500px]">
            <LogConsole logs={logs} />
          </div>
        </div>
      )}

      {view === 'vehicle' && (
        <div className="glass p-8 rounded-xl">
          <pre>{JSON.stringify(telemetry, null, 2)}</pre>
        </div>
      )}

      {view === 'rca' && (
        <div className="glass p-8 rounded-xl">
          {rca ? rca.rootCause : 'No RCA available'}
        </div>
      )}

      {view === 'ueba' && (
        <div className="glass p-8 rounded-xl">
          <LogConsole logs={logs} />
        </div>
      )}
    </div>
  );
}
