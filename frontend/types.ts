
export enum RiskBand {
  EMERGENCY = 'EMERGENCY',
  CRITICAL = 'CRITICAL',
  DEGRADED = 'DEGRADED',
  NORMAL = 'NORMAL'
}

export type ViewState = 'dashboard' | 'vehicle' | 'rca' | 'ueba';

export interface TelemetryData {
  airTemp: number;
  processTemp: number;
  rotationalSpeed: number;
  torque: number;
  toolWear: number;
}

export interface AgentLog {
  id: string;
  agent: 'Master' | 'UEBA' | 'Manufacturing';
  message: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
}

export interface RiskAnalysis {
  mlRisk: number;
  ruleRisk: number;
  finalRisk: number;
  band: RiskBand;
  decisionTrace: string[];
}

export interface RCADataset {
  rootCause: string;
  remedy: string;
  capa: string;
}

export interface UEBAAudit {
  timestamp: string;
  action: string;
  status: 'AUDITED' | 'FLAGGED';
  riskScore: number;
}
