import React from 'react';
import {
  Microscope,
  Cpu,
  Database,
  ShieldAlert,
  Layers,
  FileCheck2,
  CheckCircle2,
  BarChart2,
  Terminal,
} from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';

export const ResearchValidation: React.FC = () => {
  const { currentAnalysis } = useAnalysis();

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="terminal-card p-6 border border-[#243247] space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#243247] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <Microscope className="w-5 h-5 text-[#36C5F0]" />
              <h2 className="text-xl font-bold text-[#EDF3FA]">
                Scientific Architecture & Model Validation
              </h2>
              <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-[#36C5F0]/10 text-[#36C5F0] border border-[#36C5F0]/30">
                Peer-Review Grade
              </span>
            </div>
            <p className="text-xs text-[#91A1B5] mt-1">
              Methodological transparency, model checkpoints, data provenance, and ethical boundaries
            </p>
          </div>
        </div>

        {/* Core Model Checkpoint Specifications */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="p-4 rounded-lg bg-[#0A101C] border border-[#243247] space-y-2">
            <div className="font-bold text-[#36C5F0] uppercase tracking-wider flex items-center gap-2 text-sm">
              <Cpu className="w-4 h-4" /> 1. Computer Vision: YOLOv8n D-Fire
            </div>
            <div className="space-y-1 text-[#91A1B5] text-[11px]">
              <div><strong>Checkpoint:</strong> models/vision/fire_smoke_yolov8n_dfire_best.pt</div>
              <div><strong>Architecture:</strong> Ultralytics YOLOv8n (nano, ~3.2M params)</div>
              <div><strong>Trained Classes:</strong> 0: smoke, 1: fire</div>
              <div><strong>Inference Mode:</strong> CPU-only execution with PyTorch / OpenCV</div>
              <div><strong>IoU Suppression:</strong> Deduplication threshold &gt; 0.80</div>
              <div><strong>Plume Physics:</strong> Dark Channel Prior transmission + HSV saturation</div>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#0A101C] border border-[#243247] space-y-2">
            <div className="font-bold text-[#27C7A5] uppercase tracking-wider flex items-center gap-2 text-sm">
              <BarChart2 className="w-4 h-4" /> 2. PM2.5 Forecaster: XGBoost Baseline
            </div>
            <div className="space-y-1 text-[#91A1B5] text-[11px]">
              <div><strong>Checkpoint:</strong> models/xgboost_pm25_6h.pkl</div>
              <div><strong>Features Schema:</strong> models/baseline_features.json (exact 22 features)</div>
              <div><strong>Validated Horizon:</strong> +6 Hours ahead</div>
              <div><strong>Lags Consumed:</strong> $t-0, t-1, t-3, t-6, t-12, t-24$ (strict provenance)</div>
              <div><strong>Weather Covariates:</strong> 10 ERA5 parameters (BLH, wind, dew point, etc.)</div>
              <div><strong>Cyclical Encodings:</strong> Sine/Cosine hour-of-day & day-of-year</div>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset & Training Methodology */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="terminal-card p-5 border border-[#243247] space-y-3">
          <h3 className="text-sm font-bold text-[#EDF3FA] flex items-center gap-2">
            <Database className="w-4 h-4 text-[#F4B942]" />
            Data Provenance & Cleaning
          </h3>
          <div className="text-xs text-[#91A1B5] space-y-2">
            <p>
              <strong>Multi-Station Air Quality:</strong> 2,116,215 hourly observations across 82 Delhi monitoring stations from January 2022 to November 2026.
            </p>
            <p>
              <strong>ERA5 Weather Reanalysis:</strong> 42,504 hourly atmospheric profiles centered at Delhi coordinates (28.61°N, 77.23°E).
            </p>
            <p>
              <strong>Year 2024 Handling:</strong> Excluded from baseline model training due to station reporting outages and incomplete lag coverage across multiple key monitoring clusters.
            </p>
            <p>
              <strong>Zero Silent Interpolation:</strong> Missing station observations are never silently interpolated or synthetic-filled.
            </p>
          </div>
        </div>

        <div className="terminal-card p-5 border border-[#243247] space-y-3">
          <h3 className="text-sm font-bold text-[#EDF3FA] flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-[#F06464]" />
            Core Scientific Limitations
          </h3>
          <div className="text-xs text-[#91A1B5] space-y-2">
            <p>
              <strong>1. Optical Evidence ≠ Mass Concentration:</strong> Visual smoke bounding boxes and Dark Channel Prior plume segmentations represent optical obscuration and photometric attenuation. They do not constitute direct mass concentration (µg/m³) measurements.
            </p>
            <p>
              <strong>2. Farneback Flow ≠ Atmospheric Wind:</strong> Dense optical flow calculates apparent pixel velocity on the camera sensor plane, which can be influenced by perspective and camera jitter.
            </p>
            <p>
              <strong>3. Central Weather Grid:</strong> Weather variables reflect central Delhi ERA5 reanalysis and do not capture micro-climatic urban street-canyon variations.
            </p>
            <p>
              <strong>4. Historical Replay:</strong> Operational real-time live forecasting requires active ECMWF/GFS forecast feeds.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
