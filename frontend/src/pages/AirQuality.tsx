import React from 'react';
import {
  Wind,
  Radio,
  Clock,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Info,
  CheckCircle,
  Activity,
  Heart,
  Baby,
} from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';

export const AirQuality: React.FC = () => {
  const { currentAnalysis } = useAnalysis();

  const ns = currentAnalysis?.nearest_station;
  const aqi = currentAnalysis?.air_quality_cpcb;
  const prec = currentAnalysis?.precaution_advisory;

  const subIndices = aqi?.sub_indices || {};

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner / Assessment Card */}
      <div className="terminal-card p-6 border border-[#243247] space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#243247] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-[#EDF3FA]">
                CPCB National Air Quality Index (NAQI) Evaluation
              </h2>
              {aqi?.is_official_cpcb_compliant ? (
                <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-[#27C7A5]/10 text-[#27C7A5] border border-[#27C7A5]/30">
                  Official CPCB NAQI
                </span>
              ) : (
                <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-[#F4B942]/10 text-[#F4B942] border border-[#F4B942]/30">
                  Partial Pollutant Assessment — Not an Official CPCB AQI
                </span>
              )}
            </div>
            <p className="text-xs text-[#91A1B5] mt-1">
              Evaluated against Central Pollution Control Board (CPCB) NAQI breakpoint standards (2014/2015 Guidelines)
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-[10px] text-[#91A1B5] uppercase font-bold">Dominant Sub-Index</div>
              <div className="text-3xl font-extrabold font-mono" style={{ color: aqi?.color || '#36C5F0' }}>
                {aqi?.aqi ?? '--'}
              </div>
            </div>
            <div
              className="px-3 py-1.5 rounded-lg border font-bold text-xs"
              style={{
                backgroundColor: `${aqi?.color || '#36C5F0'}15`,
                borderColor: `${aqi?.color || '#36C5F0'}40`,
                color: aqi?.color || '#36C5F0',
              }}
            >
              {aqi?.category || 'Evaluating'}
            </div>
          </div>
        </div>

        {/* Health Impact Statement */}
        <div className="p-3.5 rounded-lg bg-[#0A101C] border border-[#243247] text-xs flex items-start gap-3">
          <Info className="w-4 h-4 text-[#36C5F0] shrink-0 mt-0.5" />
          <div>
            <strong className="text-[#EDF3FA]">CPCB Health Advisory:</strong>{' '}
            <span className="text-[#91A1B5]">
              {aqi?.health_statement || 'Breathing discomfort to people on prolonged exposure.'}
            </span>
          </div>
        </div>

        {!aqi?.is_official_cpcb_compliant && (
          <div className="p-3 rounded bg-[#F4B942]/10 border border-[#F4B942]/30 text-xs text-[#F4B942] flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <strong>CPCB 3-Pollutant Mandate Notice:</strong> Under official CPCB guidelines, overall AQI can only be declared when at least THREE criteria pollutants (with at least PM10 or PM2.5) are monitored. Since this station records particulate parameters (PM2.5 and PM10), individual sub-indices are reported without claiming a full official NAQI score.
            </div>
          </div>
        )}
      </div>

      {/* Pollutant Sub-Indices Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* PM2.5 Card */}
        <div className="terminal-card p-4 space-y-2">
          <div className="flex justify-between items-center text-xs text-[#91A1B5]">
            <span className="font-bold uppercase tracking-wider text-[10px]">PM2.5 (Fine Particulate)</span>
            <Wind className="w-4 h-4 text-[#36C5F0]" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-[#EDF3FA]">
              {subIndices.pm25 ? subIndices.pm25.concentration.toFixed(1) : (ns?.pm25_ugm3?.toFixed(1) ?? '--')}
            </span>
            <span className="text-xs text-[#91A1B5]">µg/m³</span>
          </div>
          <div className="flex justify-between items-center text-xs pt-2 border-t border-[#243247]">
            <span className="text-[#91A1B5]">Sub-Index:</span>
            <span className="font-mono font-bold text-[#F4B942]">
              {subIndices.pm25?.sub_index ?? '--'} ({subIndices.pm25?.category ?? 'Normal'})
            </span>
          </div>
        </div>

        {/* PM10 Card */}
        <div className="terminal-card p-4 space-y-2">
          <div className="flex justify-between items-center text-xs text-[#91A1B5]">
            <span className="font-bold uppercase tracking-wider text-[10px]">PM10 (Coarse Particulate)</span>
            <Wind className="w-4 h-4 text-[#27C7A5]" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-[#EDF3FA]">
              {subIndices.pm10 ? subIndices.pm10.concentration.toFixed(1) : '--'}
            </span>
            <span className="text-xs text-[#91A1B5]">µg/m³</span>
          </div>
          <div className="flex justify-between items-center text-xs pt-2 border-t border-[#243247]">
            <span className="text-[#91A1B5]">Sub-Index:</span>
            <span className="font-mono font-bold text-[#27C7A5]">
              {subIndices.pm10?.sub_index ?? '--'} ({subIndices.pm10?.category ?? '--'})
            </span>
          </div>
        </div>

        {/* Gaseous Criteria Status */}
        <div className="terminal-card p-4 space-y-2">
          <div className="flex justify-between items-center text-xs text-[#91A1B5]">
            <span className="font-bold uppercase tracking-wider text-[10px]">NO2 / SO2 / CO</span>
            <Activity className="w-4 h-4 text-[#91A1B5]" />
          </div>
          <div className="text-sm font-semibold text-[#91A1B5] mt-1">
            Gaseous Pollutants
          </div>
          <div className="text-[11px] text-[#91A1B5] pt-2 border-t border-[#243247]">
            Not Monitored at this Station
          </div>
        </div>

        {/* Station Provenance */}
        <div className="terminal-card p-4 space-y-2">
          <div className="flex justify-between items-center text-xs text-[#91A1B5]">
            <span className="font-bold uppercase tracking-wider text-[10px]">Active Station</span>
            <Radio className="w-4 h-4 text-[#36C5F0]" />
          </div>
          <div className="text-sm font-bold text-[#EDF3FA] truncate" title={ns?.station_name || 'Delhi Network'}>
            {ns?.station_name ? ns.station_name.split(',')[0] : 'Delhi Network'}
          </div>
          <div className="text-[11px] text-[#91A1B5] flex justify-between pt-2 border-t border-[#243247]">
            <span>Distance:</span>
            <span className="font-mono text-[#36C5F0]">{ns?.distance_km?.toFixed(2) ?? '--'} km</span>
          </div>
        </div>
      </div>

      {/* Precaution Advisory Breakdown */}
      {prec && (
        <div className="terminal-card p-6 border border-[#243247] space-y-4">
          <div className="flex items-center justify-between border-b border-[#243247] pb-3">
            <h3 className="text-sm font-bold uppercase tracking-wider text-[#EDF3FA] flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-[#F06464]" />
              Detailed Precaution & Public Exposure Guidelines
            </h3>
            <span className="text-xs font-mono text-[#91A1B5]">{prec.data_provenance}</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            {/* General Advice */}
            <div className="p-4 rounded-lg bg-[#0A101C] border border-[#243247] space-y-2">
              <div className="font-bold text-[#36C5F0] uppercase tracking-wider flex items-center gap-1.5">
                <CheckCircle className="w-3.5 h-3.5" /> General Public Directives
              </div>
              <ul className="space-y-1.5 list-disc list-inside text-[#EDF3FA]">
                {prec.general_advice.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>

            {/* Indoor Air Management */}
            <div className="p-4 rounded-lg bg-[#0A101C] border border-[#243247] space-y-2">
              <div className="font-bold text-[#27C7A5] uppercase tracking-wider flex items-center gap-1.5">
                <Wind className="w-3.5 h-3.5" /> Indoor Air Quality Management
              </div>
              <ul className="space-y-1.5 list-disc list-inside text-[#EDF3FA]">
                {prec.indoor_air_management.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>

            {/* Vulnerable: Children & Elderly */}
            <div className="p-4 rounded-lg bg-[#0A101C] border border-[#243247] space-y-2">
              <div className="font-bold text-[#F4B942] uppercase tracking-wider flex items-center gap-1.5">
                <Baby className="w-3.5 h-3.5" /> Children & Senior Citizens
              </div>
              <ul className="space-y-1.5 list-disc list-inside text-[#EDF3FA]">
                {prec.guidance_children_elderly.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>

            {/* Vulnerable: Respiratory & Cardiac */}
            <div className="p-4 rounded-lg bg-[#0A101C] border border-[#243247] space-y-2">
              <div className="font-bold text-[#F06464] uppercase tracking-wider flex items-center gap-1.5">
                <Heart className="w-3.5 h-3.5" /> Asthma, COPD & Cardiac Patients
              </div>
              <ul className="space-y-1.5 list-disc list-inside text-[#EDF3FA]">
                {prec.guidance_respiratory_cardiac.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Medical Disclaimer */}
          <div className="text-[11px] text-[#91A1B5] italic bg-[#0A101C] p-3 rounded border border-[#243247]">
            {prec.medical_disclaimer}
          </div>
        </div>
      )}
    </div>
  );
};
