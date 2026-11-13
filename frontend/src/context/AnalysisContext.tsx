import React, { createContext, useContext, useState, useEffect } from 'react';
import { AnalysisResponse, HealthResponse } from '../types/api';
import { checkHealth, getResultById } from '../services/api';

interface AnalysisContextType {
  currentAnalysis: AnalysisResponse | null;
  setCurrentAnalysis: (analysis: AnalysisResponse | null) => void;
  health: HealthResponse | null;
  isBackendConnected: boolean;
  recentAnalyses: AnalysisResponse[];
  addRecentAnalysis: (analysis: AnalysisResponse) => void;
  loadSampleAnalysis: (resultId: string) => Promise<void>;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisResponse[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  useEffect(() => {
    // Initial health check & periodic check
    const pollHealth = async () => {
      try {
        const data = await checkHealth();
        setHealth(data);
        setIsBackendConnected(true);
      } catch (err) {
        setIsBackendConnected(false);
      }
    };
    pollHealth();
    const timer = setInterval(pollHealth, 15000);
    return () => clearInterval(timer);
  }, []);

  const addRecentAnalysis = (analysis: AnalysisResponse) => {
    setCurrentAnalysis(analysis);
    setRecentAnalyses(prev => [analysis, ...prev.filter(a => a.result_id !== analysis.result_id)].slice(0, 10));
  };

  const loadSampleAnalysis = async (resultId: string) => {
    try {
      const data = await getResultById(resultId);
      addRecentAnalysis(data);
    } catch (err) {
      console.error('Failed to load sample analysis', err);
    }
  };

  return (
    <AnalysisContext.Provider
      value={{
        currentAnalysis,
        setCurrentAnalysis,
        health,
        isBackendConnected,
        recentAnalyses,
        addRecentAnalysis,
        loadSampleAnalysis,
        isAnalyzing,
        setIsAnalyzing,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};
