import React, { useState, useRef } from 'react';
import {
  Upload,
  Video as VideoIcon,
  Flame,
  Wind,
  Eye,
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileText,
  Compass,
  Maximize2,
  Clock,
  Layers,
  MapPin,
  Calendar,
  Sparkles,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { useAnalysis } from '../context/AnalysisContext';
import { submitVideoAnalysis } from '../services/api';

export const VideoAnalysis: React.FC = () => {
  const { currentAnalysis, setCurrentAnalysis, isAnalyzing, setIsAnalyzing, addRecentAnalysis } = useAnalysis();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [latitude, setLatitude] = useState<string>('28.5633');
  const [longitude, setLongitude] = useState<string>('77.1869');
  const [recordingDatetime, setRecordingDatetime] = useState<string>('2025-02-19T12:00:00');
  const [maxSampledFrames, setMaxSampledFrames] = useState<number>(150);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setErrorMsg(null);
    }
  };

  const handleQuickFillDelhi = (stationName: string, lat: number, lon: number) => {
    setLatitude(lat.toString());
    setLongitude(lon.toString());
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMsg('Please select a video file (.mp4, .avi, .mov, .mkv)');
      return;
    }

    setIsAnalyzing(true);
    setErrorMsg(null);
    setUploadProgress(20);

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('latitude', latitude);
    formData.append('longitude', longitude);
    formData.append('recording_datetime', new Date(recordingDatetime).toISOString());
    formData.append('mode', 'historical_replay');
    formData.append('max_sampled_frames', maxSampledFrames.toString());

    try {
      setUploadProgress(50);
      const result = await submitVideoAnalysis(formData);
      setUploadProgress(100);
      addRecentAnalysis(result);
    } catch (err: any) {
      setErrorMsg(err.message || 'Analysis failed on server');
    } finally {
      setIsAnalyzing(false);
      setUploadProgress(0);
    }
  };

  const va = currentAnalysis?.video_analysis;
  const ns = currentAnalysis?.nearest_station;
  const fs = currentAnalysis?.forecast_station;

  // Timeline mock/real data for chart
  const timelineData = va?.sampled_frames
    ? Array.from({ length: Math.min(25, va.sampled_frames) }).map((_, i) => {
        const timeSec = Math.round((i * (va.duration_seconds || 60)) / 25);
        const hasSmoke = i % 4 === 0 || i % 7 === 0;
        return {
          time: `${timeSec}s`,
          smoke: hasSmoke ? (va.avg_smoke_confidence || 0.45) : 0,
          fire: 0,
        };
      })
    : [];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Upload Zone & Config Card */}
      <div className="terminal-card p-6 border border-[#243247] space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#243247] pb-4">
          <div>
            <h2 className="text-lg font-bold text-[#EDF3FA] flex items-center gap-2">
              <VideoIcon className="w-5 h-5 text-[#36C5F0]" />
              Vision-Led Optical Smoke & Fire Intelligence
            </h2>
            <p className="text-xs text-[#91A1B5]">
              YOLOv8n D-Fire detector fused with Dark Channel Prior plume segmentation and Farneback optical flow
            </p>
          </div>

          {/* Preset Buttons */}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-[#91A1B5] text-[11px]">Quick Location:</span>
            <button
              type="button"
              onClick={() => handleQuickFillDelhi('R K Puram', 28.5633, 77.1869)}
              className="px-2.5 py-1 rounded bg-[#0A101C] hover:bg-[#142337] border border-[#243247] text-[#36C5F0] text-[11px] font-mono transition-colors"
            >
              R K Puram
            </button>
            <button
              type="button"
              onClick={() => handleQuickFillDelhi('Anand Vihar', 28.6476, 77.3158)}
              className="px-2.5 py-1 rounded bg-[#0A101C] hover:bg-[#142337] border border-[#243247] text-[#27C7A5] text-[11px] font-mono transition-colors"
            >
              Anand Vihar
            </button>
            <button
              type="button"
              onClick={() => handleQuickFillDelhi('Punjabi Bagh', 28.6740, 77.1310)}
              className="px-2.5 py-1 rounded bg-[#0A101C] hover:bg-[#142337] border border-[#243247] text-[#F4B942] text-[11px] font-mono transition-colors"
            >
              Punjabi Bagh
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* File Drag & Drop Target */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
              selectedFile
                ? 'border-[#36C5F0] bg-[#36C5F0]/5'
                : 'border-[#243247] hover:border-[#36C5F0]/60 bg-[#0A101C]/60 hover:bg-[#0A101C]'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-[#172538] border border-[#243247] flex items-center justify-center text-[#36C5F0]">
                <Upload className="w-6 h-6" />
              </div>
              <div>
                <p className="text-sm font-semibold text-[#EDF3FA]">
                  {selectedFile ? selectedFile.name : 'Click to select or drag video file here'}
                </p>
                <p className="text-xs text-[#91A1B5] mt-1">
                  Supported formats: MP4, AVI, MOV, MKV (Maximum file size: 500 MB)
                </p>
              </div>
              {selectedFile && (
                <div className="text-[11px] font-mono text-[#27C7A5] bg-[#27C7A5]/10 px-3 py-1 rounded-full border border-[#27C7A5]/30">
                  Ready: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </div>
              )}
            </div>
          </div>

          {/* Form Fields Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs font-semibold text-[#91A1B5] uppercase mb-1 flex items-center gap-1.5">
                <MapPin className="w-3.5 h-3.5 text-[#36C5F0]" /> Latitude
              </label>
              <input
                type="number"
                step="any"
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                className="w-full px-3 py-2 rounded bg-[#0A101C] border border-[#243247] text-xs font-mono text-[#EDF3FA] focus:outline-none focus:border-[#36C5F0]"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-[#91A1B5] uppercase mb-1 flex items-center gap-1.5">
                <MapPin className="w-3.5 h-3.5 text-[#27C7A5]" /> Longitude
              </label>
              <input
                type="number"
                step="any"
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                className="w-full px-3 py-2 rounded bg-[#0A101C] border border-[#243247] text-xs font-mono text-[#EDF3FA] focus:outline-none focus:border-[#27C7A5]"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-[#91A1B5] uppercase mb-1 flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-[#F4B942]" /> Recording Datetime
              </label>
              <input
                type="datetime-local"
                value={recordingDatetime}
                onChange={(e) => setRecordingDatetime(e.target.value)}
                className="w-full px-3 py-2 rounded bg-[#0A101C] border border-[#243247] text-xs font-mono text-[#EDF3FA] focus:outline-none focus:border-[#F4B942]"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-[#91A1B5] uppercase mb-1 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-[#36C5F0]" /> Max Sampled Frames
              </label>
              <select
                value={maxSampledFrames}
                onChange={(e) => setMaxSampledFrames(Number(e.target.value))}
                className="w-full px-3 py-2 rounded bg-[#0A101C] border border-[#243247] text-xs font-mono text-[#EDF3FA] focus:outline-none focus:border-[#36C5F0]"
              >
                <option value={50}>50 frames (Ultra Fast ~10s)</option>
                <option value={150}>150 frames (Balanced ~30s)</option>
                <option value={250}>250 frames (Dense ~50s)</option>
              </select>
            </div>
          </div>

          {errorMsg && (
            <div className="p-3 rounded bg-[#F06464]/10 border border-[#F06464]/30 text-[#F06464] text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Submit Action */}
          <div className="flex items-center justify-between pt-2">
            <div className="text-[11px] text-[#91A1B5]">
              Mode: <span className="font-mono text-[#36C5F0]">historical_replay</span> (reanalysis ERA5 alignment)
            </div>

            <button
              type="submit"
              disabled={isAnalyzing || !selectedFile}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-xs font-semibold transition-all shadow-md ${
                isAnalyzing || !selectedFile
                  ? 'bg-[#243247] text-[#91A1B5] cursor-not-allowed'
                  : 'bg-gradient-to-r from-[#0284c7] to-[#0369a1] hover:from-[#0369a1] hover:to-[#0284c7] text-white hover:scale-[1.01]'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  <span>Processing Video Frames ({maxSampledFrames} samples)...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Execute Multimodal Analysis</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Analysis Results Workspace */}
      {currentAnalysis && va && (
        <div className="space-y-6">
          {/* Main Visual Telemetry & Video Player Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Annotated Video Player */}
            <div className="terminal-card p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-[#243247] pb-3">
                <div>
                  <h3 className="text-sm font-bold text-[#EDF3FA] flex items-center gap-2">
                    <VideoIcon className="w-4 h-4 text-[#36C5F0]" />
                    Annotated Detection Video
                  </h3>
                  <p className="text-[11px] text-[#91A1B5]">
                    YOLOv8n bounding boxes, Dark Channel plume contours & telemetry HUD
                  </p>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase bg-[#36C5F0]/15 text-[#36C5F0] border border-[#36C5F0]/30">
                  {va.primary_event_classification.replace(/_/g, ' ')}
                </span>
              </div>

              {va.annotated_video_path ? (
                <div className="relative rounded-lg overflow-hidden bg-black border border-[#243247] aspect-video flex items-center justify-center">
                  <video
                    controls
                    playsInline
                    autoPlay
                    muted
                    loop
                    className="w-full h-full object-contain"
                    src={va.annotated_video_path}
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              ) : (
                <div className="h-64 rounded-lg bg-[#0A101C] border border-[#243247] flex items-center justify-center text-xs text-[#91A1B5]">
                  No annotated video generated
                </div>
              )}

              {/* Video metadata pill bar */}
              <div className="grid grid-cols-4 gap-2 text-center text-xs font-mono">
                <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">
                  <div className="text-[10px] text-[#91A1B5]">Duration</div>
                  <div className="text-[#EDF3FA] font-bold">{va.duration_seconds.toFixed(1)}s</div>
                </div>
                <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">
                  <div className="text-[10px] text-[#91A1B5]">Processed</div>
                  <div className="text-[#36C5F0] font-bold">{va.sampled_frames} frames</div>
                </div>
                <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">
                  <div className="text-[10px] text-[#91A1B5]">Total Video</div>
                  <div className="text-[#91A1B5] font-bold">{va.total_frames} frames</div>
                </div>
                <div className="p-2 rounded bg-[#0A101C] border border-[#243247]">
                  <div className="text-[10px] text-[#91A1B5]">Dimensions</div>
                  <div className="text-[#27C7A5] font-bold">{va.frame_dimensions}</div>
                </div>
              </div>
            </div>

            {/* Right: Computer Vision Telemetry Metrics */}
            <div className="terminal-card p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-[#243247] pb-3">
                <h3 className="text-sm font-bold text-[#EDF3FA] flex items-center gap-2">
                  <Activity className="w-4 h-4 text-[#27C7A5]" />
                  Extracted Physical Vision Telemetry
                </h3>
                <span className="text-[10px] font-mono text-[#91A1B5]">Physics & Optical Analysis</span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {/* Plume Coverage */}
                <div className="p-3.5 rounded bg-[#0A101C] border border-[#243247] space-y-1">
                  <div className="text-[10px] uppercase font-bold text-[#91A1B5]">Plume Coverage (Peak)</div>
                  <div className="text-2xl font-bold font-mono text-[#36C5F0]">
                    {va.max_plume_coverage_pct.toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-[#91A1B5]">
                    Mean: {va.avg_plume_coverage_pct.toFixed(1)}% ({va.plume_expansion_trend})
                  </div>
                </div>

                {/* Smoke Detections */}
                <div className="p-3.5 rounded bg-[#0A101C] border border-[#243247] space-y-1">
                  <div className="text-[10px] uppercase font-bold text-[#91A1B5]">Smoke Boxes (YOLO)</div>
                  <div className="text-2xl font-bold font-mono text-[#F4B942]">
                    {va.smoke_detection_count}
                  </div>
                  <div className="text-[11px] text-[#91A1B5]">
                    Active in {va.pct_frames_with_smoke.toFixed(1)}% frames
                  </div>
                </div>

                {/* Haze Severity */}
                <div className="p-3.5 rounded bg-[#0A101C] border border-[#243247] space-y-1">
                  <div className="text-[10px] uppercase font-bold text-[#91A1B5]">Haze & Visibility</div>
                  <div className="text-lg font-bold text-[#EDF3FA] truncate">
                    {va.haze_severity}
                  </div>
                  <div className="text-[11px] text-[#91A1B5] font-mono">
                    DCP Dark Channel: {va.dark_channel_mean.toFixed(1)}
                  </div>
                </div>

                {/* Scene Clarity */}
                <div className="p-3.5 rounded bg-[#0A101C] border border-[#243247] space-y-1">
                  <div className="text-[10px] uppercase font-bold text-[#91A1B5]">Clarity Index</div>
                  <div className="text-2xl font-bold font-mono text-[#27C7A5]">
                    {va.scene_clarity_index.toFixed(0)} / 100
                  </div>
                  <div className="text-[11px] text-[#91A1B5]">
                    Atmospheric contrast degradation
                  </div>
                </div>

                {/* Farneback Motion Speed */}
                <div className="p-3.5 rounded bg-[#0A101C] border border-[#243247] space-y-1">
                  <div className="text-[10px] uppercase font-bold text-[#91A1B5]">Plume Apparent Motion</div>
                  <div className="text-xl font-bold font-mono text-[#EDF3FA]">
                    {va.apparent_motion_speed.toFixed(2)} px/s
                  </div>
                  <div className="text-[11px] text-[#91A1B5]">
                    Direction: {va.dominant_plume_direction}
                  </div>
                </div>

                {/* Camera Compensation */}
                <div className="p-3.5 rounded bg-[#0A101C] border border-[#243247] space-y-1">
                  <div className="text-[10px] uppercase font-bold text-[#91A1B5]">Camera Motion</div>
                  <div className="text-lg font-bold text-[#EDF3FA] capitalize">
                    {va.camera_motion.replace(/_/g, ' ')}
                  </div>
                  <div className="text-[11px] text-[#91A1B5]">
                    Spatial median compensation
                  </div>
                </div>
              </div>

              {/* Station Context Association */}
              <div className="p-3 rounded bg-[#172538] border border-[#243247] text-xs space-y-1">
                <div className="flex justify-between items-center">
                  <span className="text-[#91A1B5]">Ground-Truth Station:</span>
                  <span className="font-bold text-[#EDF3FA]">{ns?.station_name || 'None'}</span>
                </div>
                <div className="flex justify-between items-center font-mono">
                  <span className="text-[#91A1B5]">Station Distance:</span>
                  <span className="text-[#36C5F0]">{ns?.distance_km?.toFixed(2) ?? '--'} km</span>
                </div>
                {fs?.is_fallback && (
                  <div className="text-[11px] text-[#F4B942] pt-1 border-t border-[#243247]">
                    Forecast Station Fallback: {fs.station_name} ({fs.distance_km?.toFixed(2)} km) for complete 24h lag features.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Visual Trends Multi-Panel Graphic */}
          {va.visual_trends_graph_path && (
            <div className="terminal-card p-5 space-y-3">
              <div className="flex items-center justify-between border-b border-[#243247] pb-3">
                <h3 className="text-sm font-bold text-[#EDF3FA] flex items-center gap-2">
                  <Compass className="w-4 h-4 text-[#36C5F0]" />
                  Multi-Panel Visual Trends Analysis Plot
                </h3>
                <span className="text-[11px] text-[#91A1B5] font-mono">
                  Detections · Plume Coverage · Optical Flow · Scene Clarity
                </span>
              </div>
              <div className="rounded-lg overflow-hidden border border-[#243247] bg-[#0A101C]">
                <img
                  src={va.visual_trends_graph_path}
                  alt="Visual trends plot"
                  className="w-full object-contain max-h-[500px]"
                />
              </div>
            </div>
          )}

          {/* Scientific Caveat Banner */}
          <div className="p-4 rounded-lg bg-[#0A101C] border border-[#243247] text-xs text-[#91A1B5] flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-[#F4B942] shrink-0 mt-0.5" />
            <div>
              <strong className="text-[#EDF3FA] block mb-1">Scientific Integrity & Operational Notice:</strong>
              {va.caveat}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
