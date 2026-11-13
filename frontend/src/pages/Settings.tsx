import React, { useState } from 'react';
import {
  Settings as SettingsIcon,
  Server,
  Sliders,
  CheckCircle2,
  RefreshCw,
  Cpu,
  Database,
  ExternalLink,
} from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { checkHealth } from '../services/api';

export const Settings: React.FC = () => {
  const { isBackendConnected, health } = useAnalysis();
  const [testing, setTesting] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<string | null>(null);

  const [confThreshold, setConfThreshold] = useState<number>(0.15);
  const [maxFallbackKm, setMaxFallbackKm] = useState<number>(10.0);

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await checkHealth();
      setTestResult(`Connected: ${res.api} (${res.status})`);
    } catch (err: any) {
      setTestResult(`Connection Failed: ${err.message}`);
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="terminal-card p-6 border border-[#243247] space-y-4">
        <div className="flex items-center gap-2 border-b border-[#243247] pb-4">
          <SettingsIcon className="w-5 h-5 text-[#36C5F0]" />
          <div>
            <h2 className="text-xl font-bold text-[#EDF3FA]">Platform Settings & Engine Configuration</h2>
            <p className="text-xs text-[#91A1B5]">
              Configure integration tolerances, model thresholds, and inspect backend architecture
            </p>
          </div>
        </div>

        {/* Backend Connectivity Status */}
        <div className="p-4 rounded-lg bg-[#0A101C] border border-[#243247] flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-[#121D2D] border border-[#243247] flex items-center justify-center text-[#36C5F0]">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <div className="text-sm font-semibold text-[#EDF3FA] flex items-center gap-2">
                Flask Multimodal API Server
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase border ${
                    isBackendConnected
                      ? 'bg-[#27C7A5]/10 text-[#27C7A5] border-[#27C7A5]/30'
                      : 'bg-[#F06464]/10 text-[#F06464] border-[#F06464]/30'
                  }`}
                >
                  {isBackendConnected ? 'Online' : 'Disconnected'}
                </span>
              </div>
              <div className="text-xs text-[#91A1B5] font-mono">http://127.0.0.1:5000</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {testResult && (
              <span className="text-xs font-mono text-[#27C7A5]">{testResult}</span>
            )}
            <button
              onClick={handleTestConnection}
              disabled={testing}
              className="px-4 py-2 rounded bg-[#172538] hover:bg-[#1f324c] border border-[#243247] text-xs font-semibold text-[#36C5F0] flex items-center gap-2 transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${testing ? 'animate-spin' : ''}`} />
              <span>Test Connection</span>
            </button>
          </div>
        </div>
      </div>

      {/* Model Parameter Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="terminal-card p-6 border border-[#243247] space-y-4">
          <h3 className="text-sm font-bold text-[#EDF3FA] flex items-center gap-2">
            <Sliders className="w-4 h-4 text-[#36C5F0]" />
            Vision Inference Sensitivity
          </h3>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs text-[#91A1B5] mb-1">
                <span>YOLOv8n Confidence Threshold:</span>
                <span className="font-mono text-[#36C5F0] font-bold">{confThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.50"
                step="0.01"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                className="w-full accent-[#36C5F0]"
              />
              <p className="text-[11px] text-[#91A1B5] mt-1">
                Default 0.15 is calibrated on D-Fire open-source validation benchmarks.
              </p>
            </div>

            <div>
              <div className="flex justify-between text-xs text-[#91A1B5] mb-1">
                <span>Maximum Station Fallback Radius:</span>
                <span className="font-mono text-[#27C7A5] font-bold">{maxFallbackKm.toFixed(1)} km</span>
              </div>
              <input
                type="range"
                min="3.0"
                max="30.0"
                step="1.0"
                value={maxFallbackKm}
                onChange={(e) => setMaxFallbackKm(parseFloat(e.target.value))}
                className="w-full accent-[#27C7A5]"
              />
              <p className="text-[11px] text-[#91A1B5] mt-1">
                Limits candidate stations when historical lags are incomplete at nearest station.
              </p>
            </div>
          </div>
        </div>

        {/* Runtime Environment Info */}
        <div className="terminal-card p-6 border border-[#243247] space-y-4">
          <h3 className="text-sm font-bold text-[#EDF3FA] flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[#27C7A5]" />
            Local Execution Environment
          </h3>

          <div className="space-y-2 text-xs font-mono text-[#EDF3FA]">
            <div className="flex justify-between p-2 rounded bg-[#0A101C] border border-[#243247]">
              <span className="text-[#91A1B5]">Backend OS:</span>
              <span>Windows 11 (x64)</span>
            </div>
            <div className="flex justify-between p-2 rounded bg-[#0A101C] border border-[#243247]">
              <span className="text-[#91A1B5]">Python Engine:</span>
              <span>Python 3.15 (CPU-only execution)</span>
            </div>
            <div className="flex justify-between p-2 rounded bg-[#0A101C] border border-[#243247]">
              <span className="text-[#91A1B5]">Frontend Stack:</span>
              <span>React 18 + Vite + Tailwind CSS</span>
            </div>
            <div className="flex justify-between p-2 rounded bg-[#0A101C] border border-[#243247]">
              <span className="text-[#91A1B5]">Video Codec Output:</span>
              <span>H.264 (libx264) + yuv420p</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
