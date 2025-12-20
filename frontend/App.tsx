
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  TelemetryData, 
  AgentLog, 
  RiskAnalysis, 
  RiskBand, 
  RCADataset, 
  ViewState, 
  UEBAAudit 
} from './types';
import { generateRCAFeedback } from './services/geminiService';

// --- Global UI Components ---

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
      {logs.map((log) => (
        <div key={log.id} className="border-l-2 pl-3 py-1 border-white/5 bg-white/5 rounded-r">
          <span className="text-white/20 text-[9px] block mb-0.5">{log.timestamp}</span>
          <span className={`font-bold mr-2 ${
            log.agent === 'Master' ? 'text-blue-400' : 
            log.agent === 'UEBA' ? 'text-purple-400' : 'text-emerald-400'
          }`}>
            [{log.agent}]
          </span>
          <span className={log.level === 'error' ? 'text-red-400' : log.level === 'success' ? 'text-emerald-300' : 'text-white/70'}>
            {log.message}
          </span>
        </div>
      ))}
      {logs.length === 0 && <div className="text-center text-white/10 mt-10 italic">Awaiting telemetry handshake...</div>}
    </div>
  </div>
);

// --- Main View Components ---

const DashboardView: React.FC<{
  telemetry: TelemetryData;
  updateField: (f: keyof TelemetryData, v: string) => void;
  analysis: RiskAnalysis | null;
  isLoading: boolean;
  onRun: () => void;
}> = ({ telemetry, updateField, analysis, isLoading, onRun }) => (
  <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
    <div className="lg:col-span-5 space-y-6">
      <div className="glass rounded-xl p-6">
        <h3 className="text-sm font-bold text-white mb-6 uppercase tracking-widest opacity-80 flex items-center gap-2">
          <div className="w-1.5 h-4 bg-sky-500 rounded-full"></div>
          Telemetry Inputs
        </h3>
        <div className="grid grid-cols-2 gap-4">
          {[
            { label: 'Air Temp [K]', key: 'airTemp' },
            { label: 'Process Temp [K]', key: 'processTemp' },
            { label: 'Rotation [RPM]', key: 'rotationalSpeed' },
            { label: 'Torque [Nm]', key: 'torque' },
            { label: 'Tool Wear [Min]', key: 'toolWear' }
          ].map(field => (
            <div key={field.key} className={field.key === 'toolWear' ? 'col-span-2' : ''}>
              <label className="text-[10px] font-bold text-white/30 uppercase mb-1 block">{field.label}</label>
              <input 
                type="number" 
                value={telemetry[field.key as keyof TelemetryData]} 
                onChange={e => updateField(field.key as keyof TelemetryData, e.target.value)} 
                className="w-full bg-slate-900/50 border border-white/10 rounded px-3 py-2 text-white text-sm focus:ring-1 focus:ring-sky-500 focus:outline-none"
              />
            </div>
          ))}
        </div>
        <button 
          onClick={onRun}
          disabled={isLoading}
          className="w-full mt-6 bg-sky-600 hover:bg-sky-500 py-3 rounded-lg font-bold text-white shadow-lg transition-all active:scale-[0.98] disabled:opacity-50"
        >
          {isLoading ? 'ORCHESTRATING AGENTS...' : 'SYNC & DIAGNOSE'}
        </button>
      </div>

      {analysis && (
        <div className="glass rounded-xl p-6 relative overflow-hidden">
          <div className={`absolute top-0 right-0 w-32 h-32 blur-[80px] opacity-20 bg-current ${
            analysis.band === RiskBand.EMERGENCY ? 'bg-red-500' : 'bg-sky-500'
          }`}></div>
          <h3 className="text-xs font-bold text-white/40 uppercase mb-4 tracking-widest">Fusion Risk Analytics</h3>
          <div className="text-5xl font-black mb-1 flex items-baseline gap-2">
            <span className={analysis.band === RiskBand.EMERGENCY ? 'text-red-500' : 'text-white'}>
              {analysis.finalRisk.toFixed(1)}
            </span>
            <span className="text-lg opacity-30">%</span>
          </div>
          <div className={`inline-block text-[10px] font-black px-2 py-0.5 rounded ${
            analysis.band === RiskBand.EMERGENCY ? 'bg-red-500/20 text-red-400 border border-red-500/20' : 'bg-white/10 text-white/60'
          }`}>
            {analysis.band} STATE
          </div>
          <div className="mt-6 space-y-2 border-t border-white/5 pt-4">
             {analysis.decisionTrace.map((t, i) => (
               <div key={i} className="text-[11px] text-white/50 flex gap-2">
                 <span className="text-sky-500">▶</span> {t}
               </div>
             ))}
          </div>
        </div>
      )}
    </div>

    <div className="lg:col-span-7 space-y-6 flex flex-col h-full min-h-[500px]">
       <div className="flex-1">
         <LogConsole logs={[]} /> {/* We pass real logs from parent */}
       </div>
    </div>
  </div>
);

const VehicleView: React.FC<{ telemetry: TelemetryData }> = ({ telemetry }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
    <div className="glass rounded-2xl p-8 flex flex-col items-center justify-center min-h-[400px]">
      <div className="w-48 h-48 rounded-full border-4 border-dashed border-sky-500/20 flex items-center justify-center mb-6">
        <svg className="w-24 h-24 text-sky-400 opacity-80" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 17a2 2 0 11-4 0 2 2 0 014 0zM19 17a2 2 0 11-4 0 2 2 0 014 0z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 16V6a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h1m8-1a1 1 0 01-1 1H9m4-1V8a1 1 0 011-1h2.586a1 1 0 01.707.293l3.414 3.414a1 1 0 01.293.707V16a1 1 0 01-1 1h-1m-6-1a1 1 0 001 1h1M5 17a2 2 0 104 0m-4 0a2 2 0 114 0m6 0a2 2 0 104 0m-4 0a2 2 0 114 0"></path></svg>
      </div>
      <h2 className="text-2xl font-bold">M&M XUV-700 EV</h2>
      <p className="text-white/40 text-sm">Asset ID: Mahindra-IQ-942</p>
      <div className="mt-8 flex gap-4">
        <div className="text-center">
           <p className="text-[10px] font-bold text-white/30 uppercase">Health</p>
           <p className="text-emerald-400 font-bold">94.2%</p>
        </div>
        <div className="w-px h-8 bg-white/10"></div>
        <div className="text-center">
           <p className="text-[10px] font-bold text-white/30 uppercase">Uptime</p>
           <p className="text-sky-400 font-bold">99.98%</p>
        </div>
      </div>
    </div>
    
    <div className="space-y-4">
      <div className="glass rounded-xl p-6">
        <h3 className="text-xs font-bold text-white/40 uppercase mb-4 tracking-widest">Digital Twin Status</h3>
        <div className="space-y-4">
          {Object.entries(telemetry).map(([key, val]) => (
            <div key={key} className="flex justify-between items-center border-b border-white/5 pb-2">
              <span className="text-sm text-white/60 capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
              <span className="mono text-sm font-bold text-sky-400">{val}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-6">
        <p className="text-emerald-400 font-bold text-sm mb-1">Status: Operational</p>
        <p className="text-emerald-400/60 text-xs">Edge synchronization active. Latency: 4ms.</p>
      </div>
    </div>
  </div>
);

// --- Main App ---

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
  const [rca, setRca] = useState<RCADataset | null>(null);
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [auditTrail, setAuditTrail] = useState<UEBAAudit[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
  if (!analysis) return;

  fetch("http://127.0.0.1:5000/api/ueba")
    .then(res => res.json())
    .then((data) => {
      setLogs(
        data.map((item: any, idx: number) => ({
          id: `${idx}-${item.timestamp}`,
          agent: "UEBA",
          message: item.action,
          timestamp: item.timestamp,
          level: "info"
        }))
      );
    })
    .catch(() => {
      // Fail silently – UEBA must never break UI
    });
}, [analysis]);


  const addLog = useCallback((agent: AgentLog['agent'], message: string, level: AgentLog['level'] = 'info') => {
    setLogs(prev => [{
      id: Math.random().toString(36).substr(2, 9),
      agent,
      message,
      timestamp: new Date().toLocaleTimeString(),
      level
    }, ...prev].slice(0, 50));
  }, []);
  const runDiagnostics = async () => {
  setIsLoading(true);
  addLog("Master", "Initiating agentic diagnostics...");

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(telemetry)
    });

    const data = await res.json();

    setAnalysis({
      finalRisk: data.final_risk,
      mlRisk: data.ml_risk,
      ruleRisk: data.rule_risk,
      band: data.state,
      decisionTrace: [
        "Telemetry ingested",
        "ML inference executed",
        "Rule-based thresholds evaluated",
        "Risk fusion completed",
        `System state: ${data.state}`
      ]
    });

    addLog("Master", `Risk computed: ${data.final_risk}% (${data.state})`);
  } catch (e) {
    addLog("Master", "BACKEND UNREACHABLE — SAFE MODE", "error");
  } finally {
    setIsLoading(false);
  }
};

      // 3. UEBA Log
      const audit: UEBAAudit = {
        timestamp: new Date().toISOString(),
        action: `Inference Execution (${riskResult.band})`,
        status: riskResult.finalRisk > 95 ? 'FLAGGED' : 'AUDITED',
        riskScore: riskResult.finalRisk
      };
      setAuditTrail(prev => [audit, ...prev].slice(0, 20));
      addLog('UEBA', `Decision signature logged to immutable audit trail.`);

      // 4. Manufacturing Agent (Emergency Loop)
      if (riskResult.band === RiskBand.EMERGENCY) {
        addLog('Manufacturing', `EMERGENCY ALERT: Triggering RCA Generation via LLM...`, 'warning');
        const feedback = await generateRCAFeedback(telemetry, riskResult);
        setRca(feedback);
        addLog('Manufacturing', `Root Cause identified. CAPA instructions dispatched.`, 'success');
      } else {
        setRca(null);
      }
    } catch (err) {
      addLog('Master', `CRITICAL EXCEPTION: Agentic failure. Transitioning to SAFE-MODE.`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const updateField = (field: keyof TelemetryData, val: string) => {
    const num = parseFloat(val) || 0;
    setTelemetry(prev => ({ ...prev, [field]: num }));
  };

  useEffect(() => {
  if (!analysis) return;

  fetch("/api/ueba")
    .then(res => res.json())
    .then(data => {
      setLogs(
        data.map((item: any, i: number) => ({
          id: String(i),
          agent: "UEBA",
          message: item.action,
          timestamp: item.timestamp,
          level: "info"
        }))
      );
    })
    .catch(() => {});
}, [analysis]);

useEffect(() => {
  if (analysis?.band === "EMERGENCY") {
    fetch("/api/rca")
      .then(res => res.json())
      .then(setRca);
  } else {
    setRca(null);
  }
}, [analysis]);


  return (
    <div className="min-h-screen p-4 md:p-8 bg-[#020617] text-slate-100">
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <div className="flex items-center gap-3">
             <div className="w-10 h-10 bg-sky-600 rounded-lg flex items-center justify-center shadow-lg shadow-sky-600/20">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
             </div>
             <div>
                <h1 className="text-2xl font-black tracking-tight leading-none">AutoIQ</h1>
                <p className="text-[10px] text-white/30 font-bold uppercase tracking-[0.2em] mt-1">Techathon Final Submission</p>
             </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex -space-x-2">
             <div className="w-8 h-8 rounded-full bg-blue-500 border-2 border-[#020617] flex items-center justify-center text-[10px] font-bold">M</div>
             <div className="w-8 h-8 rounded-full bg-emerald-500 border-2 border-[#020617] flex items-center justify-center text-[10px] font-bold">U</div>
             <div className="w-8 h-8 rounded-full bg-purple-500 border-2 border-[#020617] flex items-center justify-center text-[10px] font-bold">A</div>
          </div>
          <div className="h-8 w-px bg-white/10"></div>
          <span className="text-[10px] font-bold text-white/30">AGENT MESH ONLINE</span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto">
        <Navbar current={view} setView={setView} />

        {view === 'dashboard' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
             <div className="lg:col-span-8">
                <DashboardView 
                  telemetry={telemetry} 
                  updateField={updateField} 
                  analysis={analysis} 
                  isLoading={isLoading} 
                  onRun={runDiagnostics} 
                />
             </div>
             <div className="lg:col-span-4 h-[600px]">
                <LogConsole logs={logs} />
             </div>
          </div>
        )}

        {view === 'vehicle' && <VehicleView telemetry={telemetry} />}

        {view === 'rca' && (
          <div className="glass rounded-2xl p-8 max-w-4xl mx-auto">
             <h2 className="text-xl font-bold mb-6 flex items-center gap-3">
               <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
               Manufacturing RCA / CAPA Loop
             </h2>
             {rca ? (
               <div className="space-y-8">
                  <div className="grid md:grid-cols-2 gap-8">
                    <div className="space-y-1">
                      <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest">Root Cause Identified</span>
                      <p className="text-white/80 leading-relaxed text-sm bg-white/5 p-4 rounded-lg border border-white/5">{rca.rootCause}</p>
                    </div>
                    <div className="space-y-1">
                      <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest">Immediate Mitigation</span>
                      <p className="text-white/80 leading-relaxed text-sm bg-white/5 p-4 rounded-lg border border-white/5">{rca.remedy}</p>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest">CAPA (Corrective and Preventive Action)</span>
                    <div className="bg-emerald-500/5 border border-emerald-500/20 p-6 rounded-xl">
                       <p className="text-emerald-400 font-medium">{rca.capa}</p>
                    </div>
                  </div>
               </div>
             ) : (
               <div className="text-center py-20 opacity-20 italic">
                 No active RCA data. Generation occurs during EMERGENCY triggers.
               </div>
             )}
          </div>
        )}

        {view === 'ueba' && (
          <div className="glass rounded-2xl p-8 max-w-5xl mx-auto">
             <div className="flex justify-between items-center mb-8">
               <h2 className="text-xl font-bold">UEBA Behavioral Audit Trail</h2>
               <div className="flex gap-2">
                 <span className="px-3 py-1 bg-purple-500/20 text-purple-400 text-[10px] font-bold rounded-full border border-purple-500/30">AUDIT MODE ACTIVE</span>
               </div>
             </div>
             <div className="overflow-x-auto">
               <table className="w-full text-left">
                 <thead>
                   <tr className="text-[10px] font-bold text-white/20 uppercase tracking-widest border-b border-white/5">
                     <th className="pb-4">Timestamp</th>
                     <th className="pb-4">Agent Action</th>
                     <th className="pb-4">System State</th>
                     <th className="pb-4">Integrity Status</th>
                   </tr>
                 </thead>
                 <tbody className="mono text-xs">
                   {auditTrail.map((item, i) => (
                     <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors">
                       <td className="py-4 text-white/40">{new Date(item.timestamp).toLocaleTimeString()}</td>
                       <td className="py-4 font-bold text-white/70">{item.action}</td>
                       <td className="py-4">
                         <span className={item.riskScore > 75 ? 'text-red-400' : 'text-emerald-400'}>
                           Risk: {item.riskScore.toFixed(1)}%
                         </span>
                       </td>
                       <td className="py-4">
                         <span className={`px-2 py-0.5 rounded text-[10px] font-black ${
                           item.status === 'AUDITED' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                         }`}>
                           {item.status}
                         </span>
                       </td>
                     </tr>
                   ))}
                   {auditTrail.length === 0 && (
                     <tr>
                       <td colSpan={4} className="py-20 text-center text-white/10 italic">Audit log is empty. Initiate diagnostics to populate.</td>
                     </tr>
                   )}
                 </tbody>
               </table>
             </div>
          </div>
        )}
      </main>

      <footer className="max-w-7xl mx-auto mt-20 pt-8 border-t border-white/5 flex justify-between items-center text-[10px] text-white/20 font-bold uppercase tracking-widest">
         <div>&copy; AutoIQ Techathon Submission 2024</div>
         <div className="flex gap-8">
            <span>Mahindra | Hero Collaboration</span>
            <span>Agentic Architecture v2.4</span>
         </div>
      </footer>
    </div>
  );
}
