import { AnalysisResponse, HealthResponse } from '../types/api';

const API_BASE = ''; // proxied by Vite in dev to http://127.0.0.1:5000

export async function checkHealth(): Promise<HealthResponse> {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) {
    throw new Error(`Health check failed with status ${resp.status}`);
  }
  return resp.json();
}

export async function submitVideoAnalysis(formData: FormData): Promise<AnalysisResponse> {
  const resp = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || 'Video analysis failed on server');
  }
  return data;
}

export async function getResultById(resultId: string): Promise<AnalysisResponse> {
  const resp = await fetch(`${API_BASE}/results/${resultId}`);
  if (!resp.ok) {
    throw new Error(`Result ${resultId} not found`);
  }
  return resp.json();
}
