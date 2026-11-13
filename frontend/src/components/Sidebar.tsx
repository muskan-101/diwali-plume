import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Video,
  Wind,
  TrendingUp,
  Radio,
  Microscope,
  Settings,
  ChevronLeft,
  ChevronRight,
  Activity,
  ShieldCheck,
  Flame,
} from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggle }) => {
  const { isBackendConnected, health } = useAnalysis();

  const navItems = [
    { to: '/', label: 'Overview', icon: LayoutDashboard },
    { to: '/video-analysis', label: 'Video Analysis', icon: Video, badge: 'Hero' },
    { to: '/air-quality', label: 'Air Quality Intelligence', icon: Wind },
    { to: '/forecasting', label: 'Forecasting', icon: TrendingUp },
    { to: '/stations', label: 'Monitoring Stations', icon: Radio },
    { to: '/research', label: 'Research & Validation', icon: Microscope },
    { to: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside
      className={`fixed top-0 left-0 h-screen bg-[#080E19] border-r border-[#243247] transition-all duration-200 z-30 flex flex-col ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-[#243247]">
        {!collapsed ? (
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[#36C5F0]/20 to-[#27C7A5]/20 border border-[#36C5F0]/40 flex items-center justify-center text-[#36C5F0]">
              <Flame className="w-5 h-5 text-[#36C5F0]" />
            </div>
            <div>
              <div className="font-bold tracking-wider text-sm text-[#EDF3FA] flex items-center gap-2">
                DIWALI PLUME
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#36C5F0]/10 text-[#36C5F0] border border-[#36C5F0]/30 font-mono">
                  v2.0
                </span>
              </div>
              <div className="text-[10px] tracking-tight text-[#91A1B5] truncate">
                AI Air Pollution Intelligence
              </div>
            </div>
          </div>
        ) : (
          <div className="w-9 h-9 mx-auto rounded-lg bg-gradient-to-br from-[#36C5F0]/20 to-[#27C7A5]/20 border border-[#36C5F0]/40 flex items-center justify-center text-[#36C5F0]">
            <Flame className="w-5 h-5 text-[#36C5F0]" />
          </div>
        )}

        <button
          onClick={onToggle}
          className="text-[#91A1B5] hover:text-[#EDF3FA] p-1.5 rounded-md hover:bg-[#121D2D] transition-colors"
          title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-md text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-[#121D2D] text-[#36C5F0] border-l-2 border-[#36C5F0] shadow-sm'
                    : 'text-[#91A1B5] hover:text-[#EDF3FA] hover:bg-[#121D2D]/60'
                }`
              }
              title={collapsed ? item.label : undefined}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {!collapsed && (
                <div className="flex items-center justify-between w-full">
                  <span className="truncate">{item.label}</span>
                  {item.badge && (
                    <span className="text-[9px] px-1.5 py-0.2 rounded font-mono font-semibold bg-[#36C5F0]/15 text-[#36C5F0] border border-[#36C5F0]/30">
                      {item.badge}
                    </span>
                  )}
                </div>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Backend Status Footer */}
      <div className="p-3 border-t border-[#243247] bg-[#0A101C]">
        {!collapsed ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-[#91A1B5] flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-[#36C5F0]" />
                Engine Core
              </span>
              <span
                className={`inline-flex items-center gap-1 font-mono text-[10px] font-semibold ${
                  isBackendConnected ? 'text-[#27C7A5]' : 'text-[#F06464]'
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isBackendConnected ? 'bg-[#27C7A5] badge-pulse' : 'bg-[#F06464]'
                  }`}
                />
                {isBackendConnected ? 'ONLINE' : 'DISCONNECTED'}
              </span>
            </div>

            {isBackendConnected && health && (
              <div className="text-[10px] text-[#91A1B5]/75 font-mono space-y-0.5 pt-1 border-t border-[#243247]/50">
                <div className="flex justify-between">
                  <span>YOLOv8n D-Fire:</span>
                  <span className="text-[#27C7A5]">READY</span>
                </div>
                <div className="flex justify-between">
                  <span>XGBoost PM2.5 6h:</span>
                  <span className="text-[#27C7A5]">ACTIVE</span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex justify-center" title={isBackendConnected ? 'Backend Connected' : 'Disconnected'}>
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isBackendConnected ? 'bg-[#27C7A5] badge-pulse' : 'bg-[#F06464]'
              }`}
            />
          </div>
        )}
      </div>
    </aside>
  );
};
