export interface UserLocation {
  latitude: number;
  longitude: number;
}

export interface VideoAnalysisResult {
  success: boolean;
  error: string | null;
  total_frames: number;
  sampled_frames: number;
  sampling_stride: number;
  duration_seconds: number;
  fps: number;
  frame_dimensions: string;
  frames_with_smoke: number;
  frames_with_fire: number;
  smoke_detection_count: number;
  fire_detection_count: number;
  avg_smoke_confidence: number | null;
  avg_fire_confidence: number | null;
  max_confidence: number;
  pct_frames_with_smoke: number;
  pct_frames_with_fire: number;
  annotated_video_path: string | null;
  detection_csv_path: string | null;
  features_csv_path: string | null;
  visual_trends_graph_path: string | null;
  max_plume_coverage_pct: number;
  avg_plume_coverage_pct: number;
  plume_expansion_trend: string;
  dominant_plume_direction: string;
  apparent_motion_speed: number;
  camera_motion: string;
  dark_channel_mean: number;
  scene_clarity_index: number;
  haze_severity: string;
  primary_event_classification: string;
  event_confidence_score: number;
  caveat: string;
}

export interface NearestStation {
  found: boolean;
  station_id: number | null;
  station_name: string | null;
  station_latitude: number | null;
  station_longitude: number | null;
  distance_km: number | null;
  pm25_ugm3: number | null;
  reading_timestamp: string | null;
  reading_age_hours: number | null;
  pm25_sensor_count: number | null;
  data_quality: string | null;
  caveat: string;
}

export interface ForecastStation {
  station_id: number | null;
  station_name: string | null;
  distance_km: number | null;
  is_fallback: boolean;
  fallback_note: string | null;
}

export interface SubIndex {
  pollutant: string;
  concentration: number;
  unit: string;
  sub_index: number | null;
  category: string | null;
  is_valid: boolean;
}

export interface CpcbAirQuality {
  aqi: number | null;
  category: string | null;
  color: string | null;
  health_statement: string | null;
  dominant_pollutant: string | null;
  is_official_cpcb_compliant: boolean;
  status: string;
  monitored_pollutants_count: number;
  sub_indices: Record<string, SubIndex>;
  source_documentation: string;
}

export interface ForecastHorizon {
  horizon_hours: number;
  target_timestamp: string;
  predicted_pm25_ugm3: number;
  confidence_interval_90: [number, number];
  status: string;
}

export interface MultiHorizonForecast {
  available: boolean;
  forecast_station?: string;
  active_configuration?: string;
  horizons?: Record<string, ForecastHorizon>;
}

export interface Pm25Forecast6h {
  available: boolean;
  forecast_pm25_ugm3?: number;
  forecast_horizon_hours?: number;
  issue_timestamp?: string;
  target_timestamp?: string;
  data_quality?: string;
  confidence_interval_90?: [number, number];
  caveat?: string;
}

export interface PollutionEpisode {
  available: boolean;
  is_in_episode?: boolean;
  current_status?: string;
  threshold_ugm3?: number;
  current_pm25?: number;
  elapsed_duration_hours?: number;
  episode_start_time?: string | null;
  peak_pm25_observed?: number;
  peak_timestamp?: string | null;
  recent_trend?: string;
  expected_remaining_hours?: number;
  expected_improvement_time?: string | null;
  confidence_interval_hours?: [number, number];
  persistence_risk_score?: number;
  meteorological_dispersion?: string;
  projected_decay_trajectory?: Array<{ hour: number; projected_pm25: number }>;
}

export interface PrecautionAdvisory {
  alert_level: string;
  basis_of_recommendation: string;
  data_provenance: string;
  general_advice: string[];
  outdoor_guidance: string;
  respiratory_protection: string;
  indoor_air_management: string[];
  guidance_children_elderly: string[];
  guidance_respiratory_cardiac: string[];
  trigger_explainability: string;
  responsible_pollutants: string[];
  detected_visual_cues: string[];
  meteorological_stagnation_risk: string;
  medical_disclaimer: string;
}

export interface AnalysisResponse {
  result_id: string;
  mode: string;
  recording_datetime: string;
  user_location: UserLocation;
  success: boolean;
  warnings: string[];
  video_analysis: VideoAnalysisResult;
  nearest_station: NearestStation;
  forecast_station: ForecastStation;
  air_quality_cpcb: CpcbAirQuality;
  pm25_forecast_6h: Pm25Forecast6h;
  multi_horizon_forecast: MultiHorizonForecast;
  pollution_episode: PollutionEpisode;
  precaution_advisory: PrecautionAdvisory;
  system_note: string;
}

export interface HealthResponse {
  status: string;
  api: string;
  mode: string;
  vision_model_available: boolean;
  forecast_model_available: boolean;
  feature_schema_available: boolean;
  dataset_available: boolean;
  cpcb_engine_available: boolean;
  episode_engine_available: boolean;
  precaution_engine_available: boolean;
}
