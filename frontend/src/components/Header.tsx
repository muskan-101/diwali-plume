import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Upload,
  Calendar,
  MapPin,
  RefreshCw,
  AlertTriangle,
  Info,
  Server,
} from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';

export const Header: React.FC = () => {
  const { currentAnalysis, isBackendConnected, isAnalyzing } = useAnalysis();
  const location = useLocation();
  const navigate = useNavigate();

  const getPageTitle = (path: string) => {
    switch (path) {
      case '/':
        return 'Executive Overview Dashboard';
      case '/video-analysis':
        return 'Vision-Led Video Analysis Workspace';
      case '/air-quality':
        return 'Station Air Quality Intelligence';
      case '/forecasting':
        return '6-Hour Multi-Horizon Predictive Forecaster';
      case '/stations':
        return 'Delhi-NCR Monitoring Station Network';
      case '/research':
        return 'Research & Methodological Validation';
      case '/settings':
        return 'Platform Architecture & Settings';
      default:
        return 'Diwali Plume Intelligence';
    }
  };

  return (
    <header className="h-16 bg-[#080E19]/80 backdrop-blur-md border-b border-[#243247] sticky top-0 z-20 px-6 flex items-center justify-between">
      {/* Page Title & Breadcrumb */}
      <div>
        <div className="flex items-center gap-2 text-[11px] text-[#91A1B5] font-mono uppercase tracking-wider">
          <span>Diwali Plume</span>
          <span>/</span>
          <span className="text-[#36C5F0]">{location.pathname.replace('/', '') || 'Overview'}</span>
        </div>
        <h1 className="text-base font-semibold text-[#EDF3FA] tracking-tight">
          {getPageTitle(location.pathname)}
        </h1>
      </div>

      {/* Real-time Status Badges */}
      <div className="flex items-center gap-3">
        {/* Active video indicator */}
        {currentAnalysis && (
          <div className="hidden lg:flex items-center gap-3 px-3 py-1 rounded bg-[#121D2D] border border-[#243247] text-xs">
            <div className="flex items-center gap-1.5 text-[#91A1B5]">
              <MapPin className="w-3.5 h-3.5 text-[#36C5F0]" />
              <span className="font-mono text-[#EDF3FA]">
                {currentAnalysis.user_location.latitude.toFixed(4)}, {currentAnalysis.user_location.longitude.toFixed(4)}
              </span>
            </div>
            <span className="text-[#243247]">|</span>
            <div className="flex items-center gap-1.5 text-[#91A1B5]">
              <Calendar className="w-3.5 h-3.5 text-[#27C7A5]" />
              <span className="font-mono text-[#EDF3FA]">
                {new Date(currentAnalysis.recording_datetime).toISOString().slice(0, 16).replace('T', ' ')} UTC
              </span>
            </div>
          </div>
        )}

        {/* Mode badge */}
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-mono uppercase font-semibold bg-[#36C5F0]/10 text-[#36C5F0] border border-[#36C5F0]/30">
          <Server className="w-3 h-3" />
          <span>Historical Replay</span>
        </div>

        {/* Backend connectivity indicator */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-mono font-semibold border ${
            isBackendConnected
              ? 'bg-[#27C7A5]/10 text-[#27C7A5] border-[#27C7A5]/30'
              : 'bg-[#F06464]/10 text-[#F06464] border-[#F06464]/30'
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isBackendConnected ? 'bg-[#27C7A5] badge-pulse' : 'bg-[#F06464]'
            }`}
          />
          <span>{isBackendConnected ? 'API OK' : 'OFFLINE'}</span>
        </div>

        {/* Primary Action Button */}
        {location.pathname !== '/video-analysis' && (
          <button
            onClick={() => navigate('/video-analysis')}
            className="flex items-center gap-2 px-3 py-1.5 rounded bg-gradient-to-r from-[#0284c7] to-[#0369a1] hover:from-[#0369a1] hover:to-[#0284c7] text-white text-xs font-semibold shadow-sm transition-all"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Analyze Video</span>
          </button>
        )}
      </div>
    </header>
  );
};
