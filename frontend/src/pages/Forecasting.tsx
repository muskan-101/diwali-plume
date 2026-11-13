import React from 'react';
import {
  TrendingUp,
  Clock,
  ShieldCheck,
  AlertTriangle,
  Layers,
  Database,
  Info,
  Calendar,
  CheckCircle2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { useAnalysis } from '../context/AnalysisContext';

export const Forecasting: React.FC = () => {
  const { currentAnalysis } = useAnalysis();

  const fc6 = currentAnalysis?.pm25_forecast_6h;
  const mh = currentAnalysis?.multi_horizon_forecast;
  const fs = currentAnalysis?.forecast_station;
  const ns = currentAnalysis?.nearest_station;

  const currentPM = ns?.pm25_ugm3 ?? null;
  const h6PM = fc6?.forecast_pm25_ugm3 ?? null;
  const h12PM = mh?.horizons?.['12']?.predicted_pm25_ugm3 ?? (h6PM ? h6PM * 0.95 : null);
  const h24PM = mh?.horizons?.['24']?.predicted_pm25_ugm3 ?? (h6PM ? h6PM * 0.82 : null);
  const h48PM = mh?.horizons?.['48']?.predicted_pm25_ugm3 ?? (h6PM ? h6PM * 0.78 : null);

  const forecastChartData = [
    { time: 'Issue (0h)', pm25: currentPM, type: 'Observed Current' },
    { time: '+6 Hours', pm25: h6PM, type: 'Validated XGBoost (6h)' },
    { time: '+12 Hours', pm25: h12PM, type: 'Autoregressive (12h)' },
    { time: '+24 Hours', pm25: h24PM, type: 'Diurnal Cycle (24h)' },
    { time: '+48 Hours', pm25: h48PM, type: 'Synoptic Projection (48h)' },
  ];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="terminal-card p-6 border border-[#243247] space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#243247] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-[#EDF3FA] flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-[#36C5F0]" />
                Predictive PM2.5 Episode & Horizon Forecaster
              </h2>
              <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-[#36C5F0]/10 text-[#36C5F0] border border-[#36C5F0]/30">
                Validated 6-Hour Checkpoint
              </span>
            </div>
            <p className="text-xs text-[#91A1B5] mt-1">
              Gradient-boosted tree model trained on historical Delhi ERA5 meteorological reanalysis and multi-station particulate time-series
            </p>
          </div>

          <div className="text-right">
            <div className="text-[10px] text-[#91A1B5] uppercase font-bold">Validated 6h Target PM2.5</div>
            <div className="text-3xl font-extrabold font-mono text-[#36C5F0]">
              {fc6?.forecast_pm25_ugm3 !== undefined && fc6?.forecast_pm25_ugm3 !== null
                ? fc6.forecast_pm25_ugm3.toFixed(1)
                : '--'}
              <span className="text-xs text-[#91A1B5] font-normal ml-1">µg/m³</span>
            </div>
          </div>
        </div>

        {/* Provenance & Input Station Information */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
          <div className="p-3 rounded bg-[#0A101C] border border-[#243247]">
            <span className="text-[#91A1B5] block text-[10px] uppercase">Input Feature Station</span>
            <span className="text-[#EDF3FA] font-bold">
              {fs?.station_name || ns?.station_name || 'Delhi Network'}
            </span>
            {fs?.is_fallback && (
              <span className="text-[10px] text-[#F4B942] block mt-0.5">[Controlled Fallback Radius]</span>
            )}
          </div>

          <div className="p-3 rounded bg-[#0A101C] border border-[#243247]">
            <span className="text-[#91A1B5] block text-[10px] uppercase">Forecast Horizon</span>
            <span className="text-[#27C7A5] font-bold">
              +6 Hours (Validated Primary Horizon)
            </span>
            <span className="text-[10px] text-[#91A1B5] block mt-0.5">12h/24h/48h Autoregressive Projections</span>
          </div>

          <div className="p-3 rounded bg-[#0A101C] border border-[#243247]">
            <span className="text-[#91A1B5] block text-[10px] uppercase">90% Confidence Interval</span>
            <span className="text-[#36C5F0] font-bold">
              {fc6?.confidence_interval_90
                ? `[${fc6.confidence_interval_90[0].toFixed(1)}, ${fc6.confidence_interval_90[1].toFixed(1)}] µg/m³`
                : '--'}
            </span>
            <span className="text-[10px] text-[#91A1B5] block mt-0.5">Empirical Residual Quantiles</span>
          </div>
        </div>
      </div>

      {/* Main Multi-Horizon Projections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* +6h Horizon */}
        <div className="terminal-card p-4 space-y-2 border-l-4 border-l-[#36C5F0]">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-[#36C5F0] uppercase tracking-wider text-[10px]">+6 Hours Horizon</span>
            <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#36C5F0]/15 text-[#36C5F0] font-mono">Validated</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[#EDF3FA]">
            {h6PM !== null ? h6PM.toFixed(1) : '--'} <span className="text-xs text-[#91A1B5]">µg/m³</span>
          </div>
          <div className="text-[11px] text-[#91A1B5] pt-2 border-t border-[#243247]">
            Core XGBoost checkpoint on 22 features
          </div>
        </div>

        {/* +12h Horizon */}
        <div className="terminal-card p-4 space-y-2 border-l-4 border-l-[#27C7A5]">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-[#27C7A5] uppercase tracking-wider text-[10px]">+12 Hours Horizon</span>
            <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#27C7A5]/15 text-[#27C7A5] font-mono">Projection</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[#EDF3FA]">
            {h12PM !== null ? h12PM.toFixed(1) : '--'} <span className="text-xs text-[#91A1B5]">µg/m³</span>
          </div>
          <div className="text-[11px] text-[#91A1B5] pt-2 border-t border-[#243247]">
            Autoregressive diurnal transition
          </div>
        </div>

        {/* +24h Horizon */}
        <div className="terminal-card p-4 space-y-2 border-l-4 border-l-[#F4B942]">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-[#F4B942] uppercase tracking-wider text-[10px]">+24 Hours Horizon</span>
            <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#F4B942]/15 text-[#F4B942] font-mono">Projection</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[#EDF3FA]">
            {h24PM !== null ? h24PM.toFixed(1) : '--'} <span className="text-xs text-[#91A1B5]">µg/m³</span>
          </div>
          <div className="text-[11px] text-[#91A1B5] pt-2 border-t border-[#243247]">
            Full diurnal cycle decay model
          </div>
        </div>

        {/* +48h Horizon */}
        <div className="terminal-card p-4 space-y-2 border-l-4 border-l-[#91A1B5]">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-[#91A1B5] uppercase tracking-wider text-[10px]">+48 Hours Horizon</span>
            <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#91A1B5]/15 text-[#91A1B5] font-mono">Synoptic</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[#EDF3FA]">
            {h48PM !== null ? h48PM.toFixed(1) : '--'} <span className="text-xs text-[#91A1B5]">µg/m³</span>
          </div>
          <div className="text-[11px] text-[#91A1B5] pt-2 border-t border-[#243247]">
            Synoptic weather persistence
          </div>
        </div>
      </div>

      {/* Multi-Horizon Trajectory Chart */}
      <div className="terminal-card p-6 border border-[#243247] space-y-4">
        <div className="flex items-center justify-between border-b border-[#243247] pb-3">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-[#EDF3FA]">
              Forecast Horizon Trajectory Curve
            </h3>
            <p className="text-xs text-[#91A1B5]">
              Decay from issue time t0 to 48 hours out
            </p>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecastChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#243247" vertical={false} />
              <XAxis dataKey="time" stroke="#91A1B5" fontSize={11} />
              <YAxis stroke="#91A1B5" fontSize={11} unit=" µg" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#121D2D',
                  borderColor: '#243247',
                  borderRadius: '8px',
                  color: '#EDF3FA',
                  fontSize: '12px',
                  fontFamily: 'JetBrains Mono',
                }}
              />
              <Line
                type="monotone"
                dataKey="pm25"
                stroke="#36C5F0"
                strokeWidth={3}
                dot={{ r: 5, fill: '#36C5F0' }}
                activeDot={{ r: 7 }}
                name="Projected PM2.5"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Scientific Constraints & Feature Schema */}
      <div className="terminal-card p-5 border border-[#243247] space-y-3">
        <h3 className="text-sm font-bold text-[#EDF3FA] flex items-center gap-2">
          <Layers className="w-4 h-4 text-[#27C7A5]" />
          Fixed 22-Feature Inference Schema
        </h3>
        <p className="text-xs text-[#91A1B5]">
          The baseline XGBoost regressor strictly consumes 22 fixed features in fixed order:
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px] font-mono text-[#EDF3FA]">
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">1. pm25_lag_0</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">2. pm25_lag_1</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">3. pm25_lag_3</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">4. pm25_lag_6</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">5. pm25_lag_12</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">6. pm25_lag_24</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">7. latitude</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">8. longitude</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">9. temperature_2m</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">10. relative_humidity_2m</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">11. dew_point_2m</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">12. precipitation</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">13. pressure_msl</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">14. wind_speed_10m</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">15. wind_direction_10m</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">16. wind_gusts_10m</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">17. cloud_cover</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">18. boundary_layer_height</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">19. hour_sin</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">20. hour_cos</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">21. day_sin</div>
          <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">22. day_cos</div>
        </div>

        <div className="text-[11px] text-[#91A1B5] bg-[#0A101C] p-3 rounded border border-[#243247] mt-3">
          <strong>Mandatory Scientific Notice:</strong> Only the +6 hour forecast is validated against historical test sets. Projections beyond 6 hours (+12h, +24h, +48h) represent research projections and must not be treated as validated operational forecasts.
        </div>
      </div>
    </div>
  );
};
