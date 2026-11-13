import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AnalysisProvider } from './context/AnalysisContext';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Overview } from './pages/Overview';
import { VideoAnalysis } from './pages/VideoAnalysis';
import { AirQuality } from './pages/AirQuality';
import { Forecasting } from './pages/Forecasting';
import { MonitoringStations } from './pages/MonitoringStations';
import { ResearchValidation } from './pages/ResearchValidation';
import { Settings } from './pages/Settings';

export const App: React.FC = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <AnalysisProvider>
      <BrowserRouter>
        <div className="min-h-screen bg-[#0B1220] text-[#EDF3FA] flex">
          {/* Collapsible Sidebar */}
          <Sidebar
            collapsed={sidebarCollapsed}
            onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
          />

          {/* Main Content Area */}
          <div
            className={`flex-1 flex flex-col transition-all duration-200 ${
              sidebarCollapsed ? 'ml-16' : 'ml-64'
            }`}
          >
            <Header />

            <main className="flex-1 pb-16">
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/video-analysis" element={<VideoAnalysis />} />
                <Route path="/air-quality" element={<AirQuality />} />
                <Route path="/forecasting" element={<Forecasting />} />
                <Route path="/stations" element={<MonitoringStations />} />
                <Route path="/research" element={<ResearchValidation />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </AnalysisProvider>
  );
};
