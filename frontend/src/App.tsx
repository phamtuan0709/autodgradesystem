import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Activity, 
  Settings, 
  BookOpen, 
  Layers, 
  XCircle, 
  RefreshCw, 
  Sliders, 
  Info, 
  Download, 
  Shield, 
  Database, 
  Cpu,
  ChevronRight,
  TrendingUp,
  Award,
  CheckCircle2,
  AlertOctagon,
  HelpCircle,
  Edit3,
  AlignLeft,
  Settings2
} from 'lucide-react';
import type { GradingResult, Weights, BatchItem, SystemThresholds } from './types';
import { 
  DEFAULT_WEIGHTS, 
  WEIGHT_PRESETS, 
  DEFAULT_THRESHOLDS, 
  simulateGrading, 
  calculateLocalGrade 
} from './mockData';

export default function App() {
  // Navigation & Core States
  const [activeTab, setActiveTab] = useState<'dashboard' | 'single' | 'batch' | 'settings'>('dashboard');
  const [isLive, setIsLive] = useState<boolean>(false); // Default to simulated for zero-setup demo
  const [apiUrl, setApiUrl] = useState<string>('http://localhost:8000');
  const [healthStatus, setHealthStatus] = useState<{ status: string; device: string; models_loaded: boolean } | null>(null);
  
  // Weights and Thresholds (Client Local Copy, syncs with backend if Live)
  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const [activePreset, setActivePreset] = useState<string>('balanced');
  const [thresholds] = useState<SystemThresholds>(DEFAULT_THRESHOLDS);
  
  // Single Grader States
  const [singleContext, setSingleContext] = useState<string>('Photosynthesis is the process by which plants make food.');
  const [singleQuestion, setSingleQuestion] = useState<string>('Explain the process of photosynthesis.');
  const [singleReference, setSingleReference] = useState<string>('Photosynthesis converts sunlight, CO2, and water into glucose and oxygen using chlorophyll.');
  const [singleStudent, setSingleStudent] = useState<string>('Plants use sunlight and water to make food and release oxygen.');
  const [singleResult, setSingleResult] = useState<GradingResult | null>(null);
  const [isGradingSingle, setIsGradingSingle] = useState<boolean>(false);
  const [singleError, setSingleError] = useState<string | null>(null);
  
  // Batch Grader States
  const [batchQuestion, setBatchQuestion] = useState<string>('Explain the process of photosynthesis.');
  const [batchReference, setBatchReference] = useState<string>('Photosynthesis converts sunlight, CO2, and water into glucose and oxygen using chlorophyll.');
  const [batchRawInput, setBatchRawInput] = useState<string>(
    `Student A: Plants use light to make food.\nStudent B: Animals eat plants to get energy.\nStudent C: Plants use sunlight, water, carbon dioxide to produce glucose and oxygen via chlorophyll.\nStudent D: It is a process.`
  );
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);
  const [isGradingBatch, setIsGradingBatch] = useState<boolean>(false);
  const [selectedBatchItem, setSelectedBatchItem] = useState<BatchItem | null>(null);

  // Stats
  const [stats, setStats] = useState({
    totalGraded: 4,
    averageGrade: 64.63,
    correctCount: 1,
    partialCount: 2,
    incorrectCount: 1,
  });

  // Verify health on load & Live toggle
  useEffect(() => {
    if (isLive) {
      checkHealth();
    } else {
      setHealthStatus({ status: 'healthy', device: 'Client Sim (WASM)', models_loaded: true });
    }
  }, [isLive, apiUrl]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${apiUrl}/health`);
      if (res.ok) {
        const data = await res.json();
        setHealthStatus(data);
      } else {
        setHealthStatus({ status: 'offline', device: 'None', models_loaded: false });
      }
    } catch (err) {
      setHealthStatus({ status: 'offline', device: 'None', models_loaded: false });
    }
  };

  // Preset Selection Helper
  const handlePresetChange = (presetName: string) => {
    setActivePreset(presetName);
    if (WEIGHT_PRESETS[presetName]) {
      setWeights(WEIGHT_PRESETS[presetName]);
      if (singleResult) {
        recalculateSingle(WEIGHT_PRESETS[presetName]);
      }
    }
  };

  // Adjust Custom Slider Weights
  const handleWeightChange = (key: keyof Weights, value: number) => {
    setActivePreset('custom');
    const newWeights = { ...weights, [key]: Math.round(value * 100) / 100 };
    setWeights(newWeights);
    
    if (singleResult) {
      recalculateSingle(newWeights);
    }
  };

  // Recalculate Grade locally or via API
  const recalculateSingle = async (newWeights: Weights) => {
    if (!singleResult) return;
    
    if (isLive) {
      try {
        const res = await fetch(`${apiUrl}/grade/recalculate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            metrics: singleResult.metrics,
            weights: newWeights
          })
        });
        if (res.ok) {
          const data = await res.json();
          setSingleResult(prev => {
            if (!prev) return null;
            return {
              ...prev,
              metrics: {
                ...prev.metrics,
                final_grade: data.new_grade
              }
            };
          });
        }
      } catch (err) {
        const updatedGrade = calculateLocalGrade(singleResult.metrics, newWeights);
        setSingleResult(prev => {
          if (!prev) return null;
          return { ...prev, metrics: { ...prev.metrics, final_grade: updatedGrade } };
        });
      }
    } else {
      const updatedGrade = calculateLocalGrade(singleResult.metrics, newWeights);
      setSingleResult(prev => {
        if (!prev) return null;
        return { ...prev, metrics: { ...prev.metrics, final_grade: updatedGrade } };
      });
    }
  };

  // Grade Single Answer Action
  const handleGradeSingle = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGradingSingle(true);
    setSingleError(null);
    
    if (isLive) {
      try {
        const res = await fetch(`${apiUrl}/grade`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            context: singleContext,
            question: singleQuestion,
            reference: singleReference,
            student: singleStudent,
            weights: weights
          })
        });
        if (res.ok) {
          const data = await res.json();
          if (data.success) {
            setSingleResult(data.data);
            updateStats(data.data.metrics.final_grade, data.data.feedback.tags);
          } else {
            setSingleError(data.error || 'Unknown error occurred during grading.');
          }
        } else {
          const errData = await res.json().catch(() => ({}));
          setSingleError(errData.error || `Server responded with status code ${res.status}`);
        }
      } catch (err) {
        setSingleError('Failed to connect to the backend server. Verify uvicorn is running, or switch to simulated Demo Mode.');
      } finally {
        setIsGradingSingle(false);
      }
    } else {
      // Simulate
      setTimeout(() => {
        try {
          const result = simulateGrading(singleContext, singleQuestion, singleReference, singleStudent, weights);
          setSingleResult(result);
          updateStats(result.metrics.final_grade, result.feedback.tags);
        } catch (err: any) {
          setSingleError(err.message || 'Simulated grading error');
        } finally {
          setIsGradingSingle(false);
        }
      }, 800);
    }
  };

  // Update Stats Helper
  const updateStats = (score: number, tags: string[]) => {
    setStats(prev => {
      const newTotal = prev.totalGraded + 1;
      const isCorrect = tags.includes('Correct');
      const isPartial = tags.includes('Partially Correct');
      
      return {
        totalGraded: newTotal,
        averageGrade: Math.round(((prev.averageGrade * prev.totalGraded + score) / newTotal) * 100) / 100,
        correctCount: prev.correctCount + (isCorrect ? 1 : 0),
        partialCount: prev.partialCount + (isPartial ? 1 : 0),
        incorrectCount: prev.incorrectCount + (!isCorrect && !isPartial ? 1 : 0),
      };
    });
  };

  // Run Batch Grading
  const handleGradeBatch = async () => {
    if (!batchRawInput.trim()) return;
    setIsGradingBatch(true);
    setSelectedBatchItem(null);

    const lines = batchRawInput.split('\n').filter(line => line.trim().length > 0);
    const items: BatchItem[] = lines.map((line, idx) => {
      let identifier = `Student ${idx + 1}`;
      let studentAns = line;
      
      const colonIdx = line.indexOf(':');
      if (colonIdx > 0 && colonIdx < 30) {
        identifier = line.substring(0, colonIdx).trim();
        studentAns = line.substring(colonIdx + 1).trim();
      }

      return {
        id: identifier,
        context: '',
        question: batchQuestion,
        reference: batchReference,
        student: studentAns,
        status: 'grading'
      };
    });

    setBatchItems(items);

    if (isLive) {
      try {
        const batchPayload = items.map(item => ({
          context: item.context,
          question: item.question,
          reference: item.reference,
          student: item.student
        }));

        const res = await fetch(`${apiUrl}/grade/batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            items: batchPayload,
            weights: weights
          })
        });

        if (res.ok) {
          const data = await res.json();
          if (data.success && data.data.results) {
            const results = data.data.results;
            setBatchItems(prev => prev.map((item, index) => ({
              ...item,
              status: 'success',
              result: results[index]
            })));
          } else {
            throw new Error(data.message || 'Batch endpoint error');
          }
        } else {
          throw new Error(`Server returned HTTP ${res.status}`);
        }
      } catch (err: any) {
        setBatchItems(prev => prev.map(item => ({
          ...item,
          status: 'failed',
          error: err.message || 'Network error'
        })));
      } finally {
        setIsGradingBatch(false);
      }
    } else {
      for (let i = 0; i < items.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 300));
        const res = simulateGrading('', items[i].question, items[i].reference, items[i].student, weights);
        setBatchItems(prev => {
          const updated = [...prev];
          updated[i] = {
            ...updated[i],
            status: 'success',
            result: res
          };
          return updated;
        });
      }
      setIsGradingBatch(false);
    }
  };

  // Load sample datasets
  const handleLoadSample = (topic: 'photosynthesis' | 'watercycle') => {
    if (topic === 'photosynthesis') {
      setBatchQuestion('Explain the process of photosynthesis.');
      setBatchReference('Photosynthesis converts sunlight, CO2, and water into glucose and oxygen using chlorophyll.');
      setBatchRawInput(
        `Alice: Plants use sunlight, carbon dioxide, and water to make glucose (sugar) and release oxygen via chlorophyll.\n` +
        `Bob: Plants require sunlight and water to produce food. They release oxygen gas.\n` +
        `Charlie: Animals eat grass and plants to acquire chemical energy which was generated by sun.\n` +
        `David: water and sun`
      );
    } else {
      setBatchQuestion('Describe the water cycle.');
      setBatchReference('The water cycle is the continuous process where water evaporates from the Earth, condenses into clouds, and falls as precipitation.');
      setBatchRawInput(
        `John: Water evaporates from oceans, forms clouds in the sky, and then rains back down to Earth.\n` +
        `Sarah: The water cycle involves evaporation of water, condensation into clouds, and precipitation like rain.\n` +
        `Kevin: Water goes up to the sky because of heat. This happens many times.\n` +
        `Emma: Rain falls from clouds.`
      );
    }
  };

  // Export helper
  const handleExportCSV = () => {
    if (batchItems.length === 0) return;
    let csvContent = 'data:text/csv;charset=utf-8,';
    csvContent += 'Student,Answer,Grade,Correctness,Explanation\n';
    
    batchItems.forEach(item => {
      const name = item.id;
      const ans = `"${item.student.replace(/"/g, '""')}"`;
      const grade = item.result?.metrics.final_grade ?? 'N/A';
      const correctness = item.result?.feedback.tags[0] ?? 'N/A';
      const explanation = item.result ? `"${item.result.feedback.explanation.replace(/"/g, '""')}"` : 'N/A';
      csvContent += `${name},${ans},${grade},${correctness},${explanation}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `grading_batch_results.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Score circular color helper (Light Mode)
  const getScoreColor = (score: number) => {
    if (score >= 80) return 'stroke-emerald-500 text-emerald-600';
    if (score >= 55) return 'stroke-amber-500 text-amber-600';
    return 'stroke-rose-500 text-rose-600';
  };

  // Distinct Color Styling for Badges with Vibrant, Saturated Tones (Vibrant Light Mode)
  const getBadgeStyle = (tag: string) => {
    switch (tag) {
      // Correctness Levels
      case 'Correct':
        return 'bg-emerald-100/90 text-emerald-900 border-2 border-emerald-400 shadow-sm';
      case 'Partially Correct':
        return 'bg-amber-100/90 text-amber-900 border-2 border-amber-400 shadow-sm';
      case 'Incorrect':
        return 'bg-rose-100/90 text-rose-900 border-2 border-rose-400 shadow-sm';
      
      // Issue Tags (High Contrast & Distinct Colors)
      case 'Missing Concepts':
        return 'bg-sky-100/90 text-sky-900 border-2 border-sky-400 shadow-sm';
      case 'Factual Error':
        return 'bg-red-100/90 text-red-900 border-2 border-red-400 shadow-sm';
      case 'Logical Error':
        return 'bg-purple-100/90 text-purple-900 border-2 border-purple-400 shadow-sm';
      case 'Vague Expression':
        return 'bg-slate-200/90 text-slate-800 border-2 border-slate-400 shadow-sm';
      case 'Grammar Error':
        return 'bg-orange-100/90 text-orange-900 border-2 border-orange-400 shadow-sm';
      case 'Off-Topic':
        return 'bg-fuchsia-100/90 text-fuchsia-900 border-2 border-fuchsia-400 shadow-sm';
      case 'Incomplete':
        return 'bg-yellow-100/90 text-yellow-900 border-2 border-yellow-400 shadow-sm';
      default:
        return 'bg-slate-50 text-slate-700 border-2 border-slate-300';
    }
  };

  // Helper to render label next to an icon
  const getTagIcon = (tag: string) => {
    switch (tag) {
      case 'Correct':
        return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-800 flex-shrink-0" />;
      case 'Partially Correct':
        return <Info className="w-3.5 h-3.5 text-amber-800 flex-shrink-0" />;
      case 'Incorrect':
        return <XCircle className="w-3.5 h-3.5 text-rose-800 flex-shrink-0" />;
      case 'Factual Error':
      case 'Logical Error':
        return <AlertOctagon className="w-3.5 h-3.5 text-red-800 flex-shrink-0" />;
      default:
        return <HelpCircle className="w-3.5 h-3.5 text-slate-800 flex-shrink-0" />;
    }
  };

  return (
    <div className="min-h-screen bg-transparent text-slate-700 flex flex-col font-sans select-none antialiased">
      
      {/* HEADER BAR */}
      <header className="border-b border-slate-200/50 bg-white/70 backdrop-blur-md sticky top-0 z-50 px-8 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-slate-50 border border-slate-200/60 rounded-2xl shadow-sm">
            <Award className="w-5 h-5 text-indigo-650" strokeWidth={2} />
          </div>
          <div>
            <h1 className="text-base font-semibold leading-none tracking-tight text-slate-800">HybridASAG Grader</h1>
            <p className="text-xs text-slate-500 mt-1">Short Answer Evaluation Dashboard</p>
          </div>
        </div>

        {/* MODE TOGGLE */}
        <div className="flex items-center gap-6">
          <div className="flex items-center bg-slate-100/80 border border-slate-200/50 p-1 rounded-2xl">
            <button
              onClick={() => setIsLive(false)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200 ${
                !isLive 
                  ? 'bg-white text-indigo-700 shadow-sm border border-slate-200/60' 
                  : 'text-slate-500 hover:text-slate-850'
              }`}
            >
              Simulated Mode
            </button>
            <button
              onClick={() => setIsLive(true)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200 ${
                isLive 
                  ? 'bg-white text-emerald-600 shadow-sm border border-slate-200/60' 
                  : 'text-slate-500 hover:text-slate-850'
              }`}
            >
              Live API Mode
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                healthStatus?.status === 'healthy' ? 'bg-emerald-400' : 'bg-rose-400'
              }`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${
                healthStatus?.status === 'healthy' ? 'bg-emerald-500' : 'bg-rose-500'
              }`}></span>
            </span>
            <span className="text-xs text-slate-500">
              {isLive ? `Port 8000: ${healthStatus?.status || 'connecting...'}` : 'Sim Active'}
            </span>
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        
        {/* SIDEBAR NAVIGATION (Dark Slate Blue - cuts glare, divides layout) */}
        <nav className="w-64 border-r border-slate-800 bg-slate-900 p-6 flex flex-col justify-between text-slate-300">
          <div className="space-y-2">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all duration-150 ${
                activeTab === 'dashboard'
                  ? 'bg-slate-800 text-white border border-slate-700 shadow-inner'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/40 border border-transparent'
              }`}
            >
              <Activity className="w-4 h-4 text-slate-400" strokeWidth={2} />
              Overview
            </button>
            <button
              onClick={() => setActiveTab('single')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all duration-150 ${
                activeTab === 'single'
                  ? 'bg-slate-800 text-white border border-slate-700 shadow-inner'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/40 border border-transparent'
              }`}
            >
              <Play className="w-4 h-4 text-slate-400" strokeWidth={2} />
              Single Grader
            </button>
            <button
              onClick={() => setActiveTab('batch')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all duration-150 ${
                activeTab === 'batch'
                  ? 'bg-slate-800 text-white border border-slate-700 shadow-inner'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/40 border border-transparent'
              }`}
            >
              <Layers className="w-4 h-4 text-slate-400" strokeWidth={2} />
              Batch Grader
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all duration-150 ${
                activeTab === 'settings'
                  ? 'bg-slate-800 text-white border border-slate-700 shadow-inner'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/40 border border-transparent'
              }`}
            >
              <Settings className="w-4 h-4 text-slate-400" strokeWidth={2} />
              System config
            </button>
          </div>

          {/* LOWER META INFO */}
          <div className="bg-slate-950 border border-slate-800 rounded-2xl p-4 space-y-2 shadow-inner">
            <div className="flex items-center gap-2 text-slate-400">
              <Cpu className="w-3.5 h-3.5 text-slate-400" strokeWidth={1.5} />
              <span className="text-[11px]">Hardware: {healthStatus?.device || 'Detecting...'}</span>
            </div>
            <div className="flex items-center gap-2 text-slate-400">
              <Database className="w-3.5 h-3.5 text-slate-400" strokeWidth={1.5} />
              <span className="text-[11px]">Models: {healthStatus?.models_loaded ? 'Ready' : 'Unloaded'}</span>
            </div>
          </div>
        </nav>

        {/* MAIN BODY AREA */}
        <main className="flex-1 overflow-y-auto p-8 bg-slate-100">
          
          {/* TAB 1: OVERVIEW */}
          {activeTab === 'dashboard' && (
            <div className="max-w-5xl mx-auto space-y-8 animate-fade-in">
              <div className="flex items-center justify-between border-b border-slate-200 pb-4">
                <div>
                  <h2 className="text-xl font-semibold tracking-tight text-slate-800">Performance Summary</h2>
                  <p className="text-sm text-slate-500 mt-1">Aggregated scoring metrics from current session.</p>
                </div>
                {!isLive && (
                  <div className="bg-amber-50 border border-amber-250 rounded-2xl px-3 py-1.5 flex items-center gap-2 shadow-sm animate-fade-in">
                    <Info className="w-4 h-4 text-amber-600" strokeWidth={2} />
                    <span className="text-xs text-amber-700 font-semibold">Demo Mode: Grading is simulated.</span>
                  </div>
                )}
              </div>

              {/* METRICS GRID WITH GENTLE MATTE COLOR CARDS */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="bg-blue-50/90 rounded-3xl p-6 shadow-sm border-t-4 border-t-blue-500 border border-blue-200 relative overflow-hidden transition-all duration-200 hover:shadow-md">
                  <span className="text-xs text-blue-800 font-semibold uppercase tracking-wider">Total Graded</span>
                  <div className="text-3xl font-semibold text-blue-900 mt-2">{stats.totalGraded}</div>
                  <div className="text-xs text-blue-700 mt-2 flex items-center gap-1">
                    <TrendingUp className="w-3 h-3 text-blue-600" strokeWidth={2} />
                    Active evaluations
                  </div>
                </div>

                <div className="bg-indigo-50/90 rounded-3xl p-6 shadow-sm border-t-4 border-t-indigo-500 border border-indigo-200 relative overflow-hidden transition-all duration-200 hover:shadow-md">
                  <span className="text-xs text-indigo-800 font-semibold uppercase tracking-wider">Average Grade</span>
                  <div className="text-3xl font-semibold text-indigo-900 mt-2">{stats.averageGrade}%</div>
                  <div className="text-xs text-indigo-700 mt-2">Weighted average scale</div>
                </div>

                <div className="bg-emerald-50/90 rounded-3xl p-6 shadow-sm border-t-4 border-t-emerald-500 border border-emerald-200 relative overflow-hidden transition-all duration-200 hover:shadow-md">
                  <span className="text-xs text-emerald-800 font-semibold uppercase tracking-wider">Correct Answers</span>
                  <div className="text-3xl font-semibold text-emerald-900 mt-2">{stats.correctCount}</div>
                  <div className="text-xs text-emerald-700 mt-2">Pass rate: {Math.round((stats.correctCount / stats.totalGraded) * 100 || 0)}%</div>
                </div>

                <div className="bg-rose-50/90 rounded-3xl p-6 shadow-sm border-t-4 border-t-rose-500 border border-rose-200 relative overflow-hidden transition-all duration-200 hover:shadow-md">
                  <span className="text-xs text-rose-800 font-semibold uppercase tracking-wider">Logic Contradictions</span>
                  <div className="text-3xl font-semibold text-rose-900 mt-2">{stats.incorrectCount}</div>
                  <div className="text-xs text-rose-700 mt-2">Capped & penalized runs</div>
                </div>
              </div>

              {/* ARCHITECTURE DIAGRAM EXPLANATION (Vibrant Matte Cards) */}
              <div className="bg-slate-50/60 rounded-3xl p-8 shadow-sm border border-slate-200/80 space-y-6">
                <h3 className="text-sm font-semibold tracking-tight text-slate-800 flex items-center gap-2 border-b border-slate-200 pb-3">
                  <BookOpen className="w-4 h-4 text-indigo-600" strokeWidth={2} />
                  Hybrid Automated Short Answer Grading (ASAG) Pipeline
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-5 gap-6 text-xs text-slate-700">
                  <div className="p-4 bg-indigo-50/70 rounded-2xl border border-indigo-200 space-y-2 shadow-sm border-t-4 border-t-indigo-500">
                    <span className="text-indigo-950 font-semibold block">1. Semantic Similarity</span>
                    <p className="text-indigo-900/90">Uses <strong>SimCSE (RoBERTa-Large)</strong> to check core context compatibility and phrasing meaning similarity.</p>
                  </div>
                  <div className="p-4 bg-sky-50/70 rounded-2xl border border-sky-200 space-y-2 shadow-sm border-t-4 border-t-sky-500">
                    <span className="text-sky-955 font-semibold block text-sky-950">2. Keyword Coverage</span>
                    <p className="text-sky-900/90">Uses <strong>KeyBERT (MiniLM)</strong> to extract reference keywords and search for them in the student response.</p>
                  </div>
                  <div className="p-4 bg-orange-50/70 rounded-2xl border border-orange-200 space-y-2 shadow-sm border-t-4 border-t-orange-500">
                    <span className="text-orange-950 font-semibold block">3. Writing Quality</span>
                    <p className="text-orange-900/90">Uses CoLA classifier for <strong>Grammatical Acceptability</strong> and RoBERTa for <strong>Formality Score</strong>.</p>
                  </div>
                  <div className="p-4 bg-purple-50/70 rounded-2xl border border-purple-200 space-y-2 shadow-sm border-t-4 border-t-purple-500">
                    <span className="text-purple-955 font-semibold block text-purple-950">4. Logical NLI Check</span>
                    <p className="text-purple-900/90">Uses <strong>DeBERTa NLI</strong> for contradiction and entailment mapping to penalize incorrect/contradictory facts.</p>
                  </div>
                  <div className="p-4 bg-slate-100/70 rounded-2xl border border-slate-200/80 space-y-2 shadow-sm border-t-4 border-t-slate-500">
                    <span className="text-slate-900 font-semibold block">5. Reasoning Output</span>
                    <p className="text-slate-800/90">Uses <strong>Qwen2.5-3B-Instruct</strong> to parse metrics and output structured JSON explanations and suggestions.</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: SINGLE ANSWER GRADER */}
          {activeTab === 'single' && (
            <div className="max-w-6xl mx-auto space-y-8 animate-fade-in">
              <div className="flex items-center justify-between border-b border-slate-200 pb-4">
                <div>
                  <h2 className="text-xl font-semibold tracking-tight text-slate-800">Grade Student Answer</h2>
                  <p className="text-sm text-slate-500 mt-1">Run single short answer evaluation with custom criteria weights.</p>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                
                {/* INPUT FORM PANEL (Soft Blue Card with Blue Header) */}
                <form onSubmit={handleGradeSingle} className="lg:col-span-7 space-y-6">
                  
                  {/* TEXT INPUTS */}
                  <div className="bg-blue-50/70 rounded-3xl p-6 shadow-sm border border-blue-200/80 border-t-4 border-t-blue-500 space-y-4">
                    <div className="flex items-center gap-2 border-b border-blue-200/60 pb-2 mb-2">
                      <div className="p-1.5 bg-blue-100 text-blue-700 rounded-xl">
                        <AlignLeft className="w-4 h-4" strokeWidth={2.5} />
                      </div>
                      <h3 className="text-sm font-semibold text-blue-950">Grader Inputs</h3>
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-blue-900/80 font-semibold">Context (Optional)</label>
                      <input 
                        type="text" 
                        value={singleContext}
                        onChange={(e) => setSingleContext(e.target.value)}
                        placeholder="e.g. Photosynthesis converts sunlight..."
                        className="w-full text-sm bg-white border border-blue-200 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-100 transition-all duration-200 shadow-sm"
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-blue-900/80 font-semibold">Question</label>
                      <input 
                        type="text" 
                        required
                        value={singleQuestion}
                        onChange={(e) => setSingleQuestion(e.target.value)}
                        placeholder="e.g. Explain photosynthesis."
                        className="w-full text-sm bg-white border border-blue-200 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-100 transition-all duration-200 shadow-sm"
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-blue-900/80 font-semibold">Reference Answer</label>
                      <textarea 
                        required
                        rows={2}
                        value={singleReference}
                        onChange={(e) => setSingleReference(e.target.value)}
                        placeholder="Enter the golden standard response..."
                        className="w-full text-sm bg-white border border-blue-200 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-100 transition-all duration-200 resize-none shadow-sm"
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-blue-900/80 font-semibold">Student Answer</label>
                      <textarea 
                        required
                        rows={3}
                        value={singleStudent}
                        onChange={(e) => setSingleStudent(e.target.value)}
                        placeholder="Enter the student response to grade..."
                        className="w-full text-sm bg-white border border-blue-200 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-100 transition-all duration-200 resize-none font-sans shadow-sm"
                      />
                    </div>
                  </div>

                  {/* CRITERIA WEIGHTS SLIDERS (Soft Purple Card) */}
                  <div className="bg-purple-100/70 rounded-3xl p-6 shadow-sm border border-purple-200 border-t-4 border-t-purple-500 space-y-6">
                    <div className="flex items-center justify-between border-b border-purple-200/60 pb-3">
                      <div className="flex items-center gap-2">
                        <div className="p-1.5 bg-purple-200 text-purple-800 rounded-xl">
                          <Settings2 className="w-4 h-4" strokeWidth={2.5} />
                        </div>
                        <span className="text-sm font-semibold text-purple-950">Weights Configuration</span>
                      </div>
                      <select 
                        value={activePreset} 
                        onChange={(e) => handlePresetChange(e.target.value)}
                        className="text-xs bg-white border border-purple-300 text-slate-800 rounded-2xl px-3 py-2 focus:outline-none focus:border-purple-500 font-semibold shadow-sm animate-fade-in"
                      >
                        <option value="balanced">Preset: Balanced</option>
                        <option value="content_focused">Preset: Content Focused</option>
                        <option value="academic_writing">Preset: Academic Writing</option>
                        <option value="logic_heavy">Preset: Logic Heavy</option>
                        <option value="quick_check">Preset: Quick Check</option>
                        <option value="custom" disabled>Preset: Custom</option>
                      </select>
                    </div>

                    <div className="space-y-4">
                      {/* SEMANTIC */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-purple-900 font-semibold">Semantic similarity</span>
                          <span className="text-purple-950 font-semibold">{Math.round(weights.semantic * 100)}%</span>
                        </div>
                        <input 
                          type="range" min="0" max="1" step="0.05"
                          value={weights.semantic}
                          onChange={(e) => handleWeightChange('semantic', parseFloat(e.target.value))}
                          className="w-full accent-purple-600 bg-purple-200 h-1.5 rounded-2xl appearance-none cursor-pointer"
                        />
                      </div>

                      {/* COVERAGE */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-purple-900 font-semibold">Keyword Coverage</span>
                          <span className="text-purple-955 font-semibold text-purple-950">{Math.round(weights.coverage * 100)}%</span>
                        </div>
                        <input 
                          type="range" min="0" max="1" step="0.05"
                          value={weights.coverage}
                          onChange={(e) => handleWeightChange('coverage', parseFloat(e.target.value))}
                          className="w-full accent-purple-600 bg-purple-200 h-1.5 rounded-2xl appearance-none cursor-pointer"
                        />
                      </div>

                      {/* FORMALITY */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-purple-900 font-semibold">Formality (Style)</span>
                          <span className="text-purple-955 font-semibold text-purple-950">{Math.round(weights.formality * 100)}%</span>
                        </div>
                        <input 
                          type="range" min="0" max="1" step="0.05"
                          value={weights.formality}
                          onChange={(e) => handleWeightChange('formality', parseFloat(e.target.value))}
                          className="w-full accent-purple-600 bg-purple-200 h-1.5 rounded-2xl appearance-none cursor-pointer"
                        />
                      </div>

                      {/* GRAMMAR */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-purple-900 font-semibold">Grammar acceptability</span>
                          <span className="text-purple-955 font-semibold text-purple-950">{Math.round(weights.grammar * 100)}%</span>
                        </div>
                        <input 
                          type="range" min="0" max="1" step="0.05"
                          value={weights.grammar}
                          onChange={(e) => handleWeightChange('grammar', parseFloat(e.target.value))}
                          className="w-full accent-purple-600 bg-purple-200 h-1.5 rounded-2xl appearance-none cursor-pointer"
                        />
                      </div>

                      {/* LOGIC */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-purple-900 font-semibold">Logical Coherence (NLI)</span>
                          <span className="text-purple-955 font-semibold text-purple-950">{Math.round(weights.logic * 100)}%</span>
                        </div>
                        <input 
                          type="range" min="0" max="1" step="0.05"
                          value={weights.logic}
                          onChange={(e) => handleWeightChange('logic', parseFloat(e.target.value))}
                          className="w-full accent-purple-600 bg-purple-200 h-1.5 rounded-2xl appearance-none cursor-pointer"
                        />
                      </div>
                    </div>

                    <div className="text-[10px] text-purple-950 flex items-center gap-2 bg-purple-200/50 p-3 rounded-2xl border border-purple-300/40">
                      <Info className="w-4 h-4 text-purple-800 flex-shrink-0" strokeWidth={1.5} />
                      <span>Input weights normalize dynamically to total 100% when evaluating. Recalculation is computed instantly client-side on slider adjustments.</span>
                    </div>
                  </div>

                  {singleError && (
                    <div className="bg-rose-100/80 border border-rose-250 rounded-2xl p-4 flex gap-3 text-xs text-rose-700 shadow-sm">
                      <XCircle className="w-4 h-4 flex-shrink-0" strokeWidth={2} />
                      <span>{singleError}</span>
                    </div>
                  )}

                  <button 
                    type="submit"
                    disabled={isGradingSingle}
                    className="w-full flex items-center justify-center gap-2 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-2xl font-semibold text-sm transition-all duration-200 disabled:opacity-50 shadow-sm"
                  >
                    {isGradingSingle ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" strokeWidth={2.5} />
                        Evaluating...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" strokeWidth={2.5} />
                        Run Evaluator
                      </>
                    )}
                  </button>
                </form>

                {/* RESULTS VIEW PANEL */}
                <div className="lg:col-span-5 space-y-6">
                  {singleResult ? (
                    <div className="space-y-6 animate-fade-in">
                      
                      {/* SCORE CARD */}
                      <div className={`rounded-3xl p-6 shadow-sm border ${
                        singleResult.metrics.final_grade >= 80 
                          ? 'bg-emerald-50/95 border-emerald-300 border-t-4 border-t-emerald-500 text-emerald-950' 
                          : singleResult.metrics.final_grade >= 55 
                          ? 'bg-amber-50/95 border-amber-300 border-t-4 border-t-amber-500 text-amber-950' 
                          : 'bg-rose-50/95 border-rose-300 border-t-4 border-t-rose-500 text-rose-950'
                      } flex items-center justify-between relative overflow-hidden transition-all duration-200 hover:shadow-md`}>
                        <div className="space-y-2">
                          <span className={`text-xs uppercase tracking-wider font-semibold ${
                            singleResult.metrics.final_grade >= 80 
                              ? 'text-emerald-800' 
                              : singleResult.metrics.final_grade >= 55 
                              ? 'text-amber-800' 
                              : 'text-rose-800'
                          }`}>Evaluation Grade</span>
                          <div className="flex items-baseline gap-1">
                            <span className="text-4xl font-semibold tracking-tight">{singleResult.metrics.final_grade}</span>
                            <span className={`text-sm ${
                              singleResult.metrics.final_grade >= 80 
                                ? 'text-emerald-700/80' 
                                : singleResult.metrics.final_grade >= 55 
                                ? 'text-amber-700/80' 
                                : 'text-rose-700/80'
                            }`}>/ 100</span>
                          </div>
                          
                          {/* Saturated Tags Display */}
                          <div className="flex flex-wrap gap-1.5 mt-3">
                            {singleResult.feedback.tags.map((tag, idx) => (
                              <span 
                                key={idx} 
                                className={`text-[10px] px-2.5 py-0.8 rounded-lg border flex items-center gap-1.5 font-semibold ${getBadgeStyle(tag)}`}
                              >
                                {getTagIcon(tag)}
                                {tag}
                              </span>
                            ))}
                          </div>
                        </div>

                        {/* Circular Score Visualizer */}
                        <div className="relative w-20 h-20">
                          <svg className="w-full h-full" viewBox="0 0 36 36">
                            <path
                              className={singleResult.metrics.final_grade >= 80 
                                ? 'stroke-emerald-100/50' 
                                : singleResult.metrics.final_grade >= 55 
                                ? 'stroke-amber-100/50' 
                                : 'stroke-rose-100/50'}
                              strokeWidth="2.5"
                              fill="none"
                              d="M18 2.0845
                                a 15.9155 15.9155 0 0 1 0 31.831
                                a 15.9155 15.9155 0 0 1 0 -31.831"
                            />
                            <path
                              className={`progress-circle ${getScoreColor(singleResult.metrics.final_grade).split(' ')[0]}`}
                              strokeWidth="2.5"
                              strokeLinecap="round"
                              fill="none"
                              style={{ '--offset': 283 - (283 * singleResult.metrics.final_grade) / 100 } as any}
                              d="M18 2.0845
                                a 15.9155 15.9155 0 0 1 0 31.831
                                a 15.9155 15.9155 0 0 1 0 -31.831"
                            />
                          </svg>
                          <div className={`absolute inset-0 flex items-center justify-center text-xs font-semibold ${getScoreColor(singleResult.metrics.final_grade).split(' ')[1]}`}>
                            {Math.round(singleResult.metrics.final_grade)}%
                          </div>
                        </div>
                      </div>

                      {/* LLM FEEDBACK CARDS (Soft Violet Card) */}
                      <div className="bg-violet-50/80 rounded-3xl p-6 shadow-sm border border-violet-200 border-t-4 border-t-violet-500 space-y-6">
                        <div className="space-y-2 border-b border-violet-200 pb-4">
                          <span className="text-xs text-violet-850 font-semibold block uppercase tracking-wider">Explanation</span>
                          <p className="text-xs text-violet-950 leading-relaxed font-normal">
                            {singleResult.feedback.explanation}
                          </p>
                        </div>
                        
                        <div className="space-y-2">
                          <span className="text-xs text-violet-850 font-semibold block uppercase tracking-wider">Recommendation & Actionable Steps</span>
                          <p className="text-xs text-violet-955 leading-relaxed font-normal">
                            {singleResult.feedback.suggestion}
                          </p>
                        </div>
                      </div>

                      {/* TECHNICAL BREAKDOWN SECTION (Grid of beautiful distinct Emerald-tinted cards) */}
                      <div className="bg-emerald-50/80 rounded-3xl p-6 shadow-sm border border-emerald-200 border-t-4 border-t-emerald-500 space-y-4">
                        <span className="text-xs text-emerald-900 font-semibold block border-b border-emerald-200 pb-2 uppercase tracking-wider">Technical Feature Breakdown</span>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                          {/* Semantic Similarity */}
                          <div className="p-3 bg-blue-100/60 rounded-2xl border border-blue-200/80 flex items-center justify-between shadow-sm hover:shadow transition-shadow">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                              <span className="text-blue-900 font-semibold">Semantic</span>
                            </div>
                            <span className="text-blue-950 font-semibold">{Math.round(singleResult.metrics.semantic_score * 100)}%</span>
                          </div>

                          {/* Keyword Coverage */}
                          <div className="p-3 bg-sky-100/60 rounded-2xl border border-sky-200/80 flex items-center justify-between shadow-sm hover:shadow transition-shadow">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full bg-sky-500" />
                              <span className="text-sky-900 font-semibold">Keywords</span>
                            </div>
                            <span className="text-sky-950 font-semibold">{Math.round(singleResult.metrics.coverage_score * 100)}%</span>
                          </div>

                          {/* Formality */}
                          <div className="p-3 bg-slate-100/80 rounded-2xl border border-slate-200/80 flex items-center justify-between shadow-sm hover:shadow transition-shadow">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
                              <span className="text-slate-650 font-semibold">Formality</span>
                            </div>
                            <span className="text-slate-800 font-semibold">{Math.round(singleResult.metrics.formality_score * 100)}%</span>
                          </div>

                          {/* Grammar */}
                          <div className="p-3 bg-orange-100/60 rounded-2xl border border-orange-200/80 flex items-center justify-between shadow-sm hover:shadow transition-shadow">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full bg-orange-500" />
                              <span className="text-orange-900 font-semibold">Grammar</span>
                            </div>
                            <span className="text-orange-955 font-semibold">{Math.round(singleResult.metrics.grammar_score * 100)}%</span>
                          </div>

                          {/* Logic/NLI Details */}
                          <div className="p-3 bg-purple-100/60 rounded-2xl border border-purple-200/80 flex items-center justify-between shadow-sm hover:shadow transition-shadow md:col-span-2">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
                              <span className="text-purple-900 font-semibold">Logical Consistency</span>
                            </div>
                            <span className="text-purple-955 font-semibold">{Math.round(singleResult.metrics.logic_score * 100)}%</span>
                          </div>
                        </div>

                        {/* Missing Keywords display block */}
                        {singleResult.metrics.missing_keywords && singleResult.metrics.missing_keywords.length > 0 && (
                          <div className="bg-rose-100/40 p-3 rounded-2xl border border-rose-200 mt-2 space-y-1.5 shadow-inner">
                            <span className="text-[10px] text-rose-800 font-semibold block uppercase tracking-wider">Missing Reference Keywords:</span>
                            <div className="flex flex-wrap gap-1.5">
                              {singleResult.metrics.missing_keywords.map((kw, idx) => (
                                <span key={idx} className="bg-rose-100 text-rose-900 border border-rose-250 text-[10px] px-2.5 py-0.5 rounded-lg font-semibold">
                                  {kw}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* NLI probabilities breakdown */}
                        {singleResult.metrics.logic_details && (
                          <div className="bg-emerald-50/40 p-3.5 rounded-2xl border border-emerald-200/60 text-[10px] text-emerald-950 space-y-1.5 shadow-inner">
                            <span className="font-semibold text-emerald-900 uppercase tracking-wider block">Natural Language Inference (NLI) Details:</span>
                            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-emerald-800">
                              <div>Contradiction: {Math.round((singleResult.metrics.logic_details.contradiction || 0) * 100)}%</div>
                              <div>Entailment: {Math.round((singleResult.metrics.logic_details.entailment || 0) * 100)}%</div>
                              <div>Backward-Contradiction: {Math.round((singleResult.metrics.logic_details.backward_contradiction || 0) * 100)}%</div>
                              <div>Backward-Entailment: {Math.round((singleResult.metrics.logic_details.backward_entailment || 0) * 100)}%</div>
                            </div>
                          </div>
                        )}
                      </div>

                    </div>
                  ) : (
                    <div className="h-full bg-slate-50 rounded-3xl p-8 flex flex-col items-center justify-center text-center space-y-3 min-h-[300px] border border-slate-200 shadow-sm">
                      <div className="p-3 bg-white border border-slate-200 rounded-full shadow-sm">
                        <Sliders className="w-5 h-5 text-slate-400" strokeWidth={1.5} />
                      </div>
                      <h3 className="text-sm font-semibold text-slate-800">Ready for Evaluation</h3>
                      <p className="text-xs text-slate-500 max-w-[240px] leading-relaxed">
                        Submit a student response to review model scores, tags, and AI-generated feedback.
                      </p>
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* TAB 3: BATCH GRADER */}
          {activeTab === 'batch' && (
            <div className="max-w-6xl mx-auto space-y-8 animate-fade-in">
              <div className="flex items-center justify-between border-b border-slate-200 pb-4">
                <div>
                  <h2 className="text-xl font-semibold tracking-tight text-slate-800">Batch Evaluation</h2>
                  <p className="text-sm text-slate-500 mt-1">Grade multiple student responses at once using pre-loaded templates or raw lists.</p>
                </div>
                
                {/* Sample datasets loading */}
                <div className="flex gap-2">
                  <button 
                    onClick={() => handleLoadSample('photosynthesis')}
                    className="text-xs bg-white hover:bg-emerald-50 hover:text-emerald-700 text-slate-700 px-3.5 py-2 rounded-2xl border border-slate-200 transition-all font-semibold shadow-sm cursor-pointer"
                  >
                    Template: Photosynthesis
                  </button>
                  <button 
                    onClick={() => handleLoadSample('watercycle')}
                    className="text-xs bg-white hover:bg-sky-50 hover:text-sky-700 text-slate-700 px-3.5 py-2 rounded-2xl border border-slate-200 transition-all font-semibold shadow-sm cursor-pointer"
                  >
                    Template: Water Cycle
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                
                {/* INPUT AND BATCH PARSER */}
                <div className="lg:col-span-5 space-y-6">
                  <div className="bg-blue-50/70 rounded-3xl p-6 shadow-sm border border-blue-200/80 border-t-4 border-t-blue-500 space-y-4">
                    <div className="flex items-center gap-2 border-b border-blue-200/60 pb-2 mb-2">
                      <div className="p-1.5 bg-blue-100 text-blue-700 rounded-lg">
                        <Edit3 className="w-4 h-4" strokeWidth={2.5} />
                      </div>
                      <h3 className="text-sm font-semibold text-blue-950">Batch Configuration</h3>
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-blue-900/80 font-semibold">Question Title</label>
                      <input 
                        type="text" 
                        value={batchQuestion}
                        onChange={(e) => setBatchQuestion(e.target.value)}
                        className="w-full text-sm bg-white border border-blue-200 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-blue-500 focus:bg-white transition-all font-sans shadow-sm"
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-blue-900/80 font-semibold">Golden Reference Answer</label>
                      <textarea 
                        rows={2}
                        value={batchReference}
                        onChange={(e) => setBatchReference(e.target.value)}
                        className="w-full text-sm bg-white border border-blue-200 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-blue-500 focus:bg-white transition-all resize-none font-sans shadow-sm"
                      />
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between items-center">
                        <label className="text-xs text-blue-900/80 font-semibold">Student Answers (Format: Identifier: Answer, one per line)</label>
                      </div>
                      <textarea 
                        rows={6}
                        value={batchRawInput}
                        onChange={(e) => setBatchRawInput(e.target.value)}
                        placeholder="e.g. Student A: Plants absorb sunlight..."
                        className="w-full text-xs bg-white border border-blue-200 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-blue-500 focus:bg-white transition-all resize-none font-mono shadow-sm"
                      />
                    </div>

                    <button
                      onClick={handleGradeBatch}
                      disabled={isGradingBatch || !batchRawInput.trim()}
                      className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-2xl flex items-center justify-center gap-2 transition-all shadow-sm cursor-pointer disabled:opacity-50"
                    >
                      {isGradingBatch ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" strokeWidth={2.5} />
                          Batch grading in progress...
                        </>
                      ) : (
                        <>
                          <Play className="w-3.5 h-3.5" strokeWidth={2.5} />
                          Evaluate Batch
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* BATCH TABLE & DETAILS PANEL */}
                <div className="lg:col-span-7 space-y-6">
                  {batchItems.length > 0 ? (
                    <div className="space-y-6 animate-fade-in">
                      
                      {/* BATCH CONTROL & ACTIONS */}
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-500 font-semibold">Evaluation Status ({batchItems.length} records)</span>
                        <button
                          onClick={handleExportCSV}
                          className="text-xs text-slate-700 hover:text-slate-900 hover:bg-slate-100 flex items-center gap-1.5 bg-white border border-slate-200 px-3.5 py-2 rounded-2xl transition-all shadow-sm cursor-pointer"
                        >
                          <Download className="w-3.5 h-3.5" strokeWidth={1.5} />
                          Export CSV
                        </button>
                      </div>

                      {/* DATA TABLE (Dynamic Colored Rows based on status) */}
                      <div className="bg-slate-50/80 rounded-3xl overflow-hidden border border-slate-200/80 border-t-4 border-t-amber-500 shadow-sm">
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="border-b border-slate-200 bg-slate-100/60 text-slate-500">
                                <th className="p-4 font-semibold">Identifier</th>
                                <th className="p-4 font-semibold">Student Answer</th>
                                <th className="p-4 font-semibold text-right">Grade</th>
                                <th className="p-4 font-semibold">Correctness</th>
                                <th className="p-4 font-semibold"></th>
                              </tr>
                            </thead>
                            <tbody>
                              {batchItems.map((item) => {
                                const finalGrade = item.result?.metrics.final_grade;
                                const correctnessTag = item.result?.feedback.tags[0] || 'Unknown';
                                
                                // Dynamic pastel backgrounds for table rows
                                let rowBg = 'bg-white hover:bg-slate-50';
                                if (item.status === 'success' && finalGrade !== undefined) {
                                  if (finalGrade >= 80) {
                                    rowBg = 'bg-emerald-50/40 hover:bg-emerald-100/50';
                                  } else if (finalGrade >= 55) {
                                    rowBg = 'bg-amber-50/40 hover:bg-amber-100/50';
                                  } else {
                                    rowBg = 'bg-rose-50/40 hover:bg-rose-100/50';
                                  }
                                } else if (item.status === 'failed') {
                                  rowBg = 'bg-rose-100/30 hover:bg-rose-100/50';
                                }

                                const isSelected = selectedBatchItem?.id === item.id;

                                return (
                                  <tr 
                                    key={item.id} 
                                    className={`border-b border-slate-200/60 transition-all ${rowBg} ${
                                      isSelected ? 'ring-2 ring-indigo-500/50 border-l-4 border-l-indigo-650' : ''
                                    }`}
                                  >
                                    <td className="p-4 text-slate-800 font-semibold">{item.id}</td>
                                    <td className="p-4 text-slate-500 max-w-[200px] truncate">{item.student}</td>
                                    <td className="p-4 text-right font-semibold text-slate-800">
                                      {item.status === 'grading' ? (
                                        <RefreshCw className="w-3.5 h-3.5 animate-spin ml-auto text-slate-400" />
                                      ) : item.status === 'success' ? (
                                        `${finalGrade}%`
                                      ) : (
                                        <span className="text-rose-600 font-semibold">Err</span>
                                      )}
                                    </td>
                                    <td className="p-4">
                                      {item.status === 'success' && (
                                        <span className={`px-2.5 py-0.8 rounded-lg border text-[9px] font-semibold flex items-center gap-1 w-max ${getBadgeStyle(correctnessTag)}`}>
                                          {getTagIcon(correctnessTag)}
                                          {correctnessTag}
                                        </span>
                                      )}
                                    </td>
                                    <td className="p-4 text-right">
                                      <button
                                        onClick={() => setSelectedBatchItem(item)}
                                        className="text-[10px] text-slate-500 hover:text-indigo-600 font-semibold flex items-center gap-0.5 ml-auto cursor-pointer"
                                      >
                                        Details
                                        <ChevronRight className="w-3 h-3" strokeWidth={2.5} />
                                      </button>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* SELECTED STUDENT DETAIL DRAWER (Emerald Accent with Vibrant cards) */}
                      {selectedBatchItem?.result && (
                        <div className="bg-emerald-50/70 rounded-3xl p-6 shadow-sm border border-emerald-200 border-t-4 border-t-emerald-500 space-y-4 animate-fade-in">
                          <div className="flex items-center justify-between border-b border-emerald-200/60 pb-3">
                            <div>
                              <h4 className="text-xs text-emerald-800 font-semibold uppercase tracking-wider">Detailed Evaluation View</h4>
                              <p className="text-xs text-emerald-950 font-semibold mt-1 font-sans">
                                {selectedBatchItem.student}
                              </p>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`px-2.5 py-0.8 rounded-lg border text-[9px] font-semibold flex items-center gap-1.5 ${getBadgeStyle(selectedBatchItem.result.feedback.tags[0])}`}>
                                {getTagIcon(selectedBatchItem.result.feedback.tags[0])}
                                {selectedBatchItem.result.feedback.tags[0]}
                              </span>
                              <span className="text-sm font-semibold text-emerald-950">{selectedBatchItem.result.metrics.final_grade}%</span>
                            </div>
                          </div>

                          <div className="grid grid-cols-2 gap-4 text-xs">
                            <div className="space-y-1">
                              <span className="text-emerald-900 font-semibold block uppercase tracking-wider text-[10px]">Explanation</span>
                              <p className="text-emerald-950 font-normal leading-relaxed text-[11px] bg-emerald-50/30 p-3 rounded-2xl border border-emerald-100/60 shadow-sm">
                                {selectedBatchItem.result.feedback.explanation}
                              </p>
                            </div>
                            <div className="space-y-1">
                              <span className="text-emerald-900 font-semibold block uppercase tracking-wider text-[10px]">Recommendation</span>
                              <p className="text-emerald-955 font-normal leading-relaxed text-[11px] bg-emerald-50/30 p-3 rounded-2xl border border-emerald-100/60 shadow-sm">
                                {selectedBatchItem.result.feedback.suggestion}
                              </p>
                            </div>
                          </div>

                          <div className="grid grid-cols-5 gap-2.5 text-center pt-3 border-t border-emerald-200/60 text-[10px]">
                            <div className="p-2.5 bg-blue-100/60 rounded-2xl border border-blue-200/80 shadow-sm">
                              <span className="block text-blue-900 font-semibold">Semantic</span>
                              <span className="block text-blue-950 font-semibold mt-1">{Math.round(selectedBatchItem.result.metrics.semantic_score * 100)}%</span>
                            </div>
                            <div className="p-2.5 bg-sky-100/60 rounded-2xl border border-sky-200/80 shadow-sm">
                              <span className="block text-sky-900 font-semibold">Coverage</span>
                              <span className="block text-sky-955 font-semibold mt-1">{Math.round(selectedBatchItem.result.metrics.coverage_score * 100)}%</span>
                            </div>
                            <div className="p-2.5 bg-slate-100/80 rounded-2xl border border-slate-200/80 shadow-sm">
                              <span className="block text-slate-600 font-semibold">Formality</span>
                              <span className="block text-slate-800 font-semibold mt-1">{Math.round(selectedBatchItem.result.metrics.formality_score * 100)}%</span>
                            </div>
                            <div className="p-2.5 bg-orange-100/60 rounded-2xl border border-orange-200/80 shadow-sm">
                              <span className="block text-orange-900 font-semibold">Grammar</span>
                              <span className="block text-orange-955 font-semibold mt-1">{Math.round(selectedBatchItem.result.metrics.grammar_score * 100)}%</span>
                            </div>
                            <div className="p-2.5 bg-purple-100/60 rounded-2xl border border-purple-200/80 shadow-sm">
                              <span className="block text-purple-900 font-semibold">Logic NLI</span>
                              <span className="block text-purple-955 font-semibold mt-1">{Math.round(selectedBatchItem.result.metrics.logic_score * 100)}%</span>
                            </div>
                          </div>
                        </div>
                      )}

                    </div>
                  ) : (
                    <div className="bg-slate-50 rounded-3xl p-8 flex flex-col items-center justify-center text-center space-y-3 min-h-[300px] border border-slate-200 shadow-sm">
                      <div className="p-3 bg-white border border-slate-200 rounded-full shadow-sm">
                        <Layers className="w-5 h-5 text-slate-400" strokeWidth={1.5} />
                      </div>
                      <h3 className="text-sm font-semibold text-slate-800">No batch runs yet</h3>
                      <p className="text-xs text-slate-500 max-w-[240px] leading-relaxed">
                        Load a template or enter lists of answers on the left to evaluate all students in batch.
                      </p>
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* TAB 4: SYSTEM CONFIG */}
          {activeTab === 'settings' && (
            <div className="max-w-3xl mx-auto space-y-8 animate-fade-in">
              <div className="flex items-center justify-between border-b border-slate-200 pb-4">
                <div>
                  <h2 className="text-xl font-semibold tracking-tight text-slate-800">System Configuration</h2>
                  <p className="text-sm text-slate-500 mt-1">Configure backend endpoints and view active scoring thresholds.</p>
                </div>
              </div>

              {/* ENDPOINT CONFIG */}
              <div className="bg-indigo-50/70 rounded-3xl p-6 shadow-sm border border-indigo-200 border-t-4 border-t-indigo-500 space-y-4">
                <div className="flex items-center gap-2 border-b border-indigo-200/60 pb-2 mb-2">
                  <div className="p-1.5 bg-indigo-100 text-indigo-700 rounded-lg">
                    <Database className="w-4 h-4" strokeWidth={2.5} />
                  </div>
                  <h3 className="text-sm font-semibold text-indigo-950">FastAPI Connection Settings</h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                  <div className="md:col-span-3 space-y-1">
                    <label className="text-[10px] text-indigo-900/80">Server Endpoint URL</label>
                    <input 
                      type="text" 
                      value={apiUrl}
                      onChange={(e) => setApiUrl(e.target.value)}
                      className="w-full text-xs bg-white border border-indigo-200 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none focus:border-indigo-500 focus:bg-white font-mono shadow-sm"
                    />
                  </div>
                  <button
                    onClick={checkHealth}
                    className="w-full py-2.5 bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 font-semibold text-xs rounded-2xl flex items-center justify-center gap-1.5 transition-all shadow-sm cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5 text-slate-500 animate-spin-hover" strokeWidth={2} />
                    Test Connection
                  </button>
                </div>
              </div>

              {/* SYSTEM THRESHOLDS INFO */}
              <div className="bg-purple-100/70 rounded-3xl p-6 shadow-sm border border-purple-200 border-t-4 border-t-purple-500 space-y-4">
                <div className="flex items-center justify-between border-b border-purple-200/60 pb-3">
                  <h3 className="text-sm font-semibold text-purple-950 flex items-center gap-2">
                    <Shield className="w-4 h-4 text-purple-800" strokeWidth={1.5} />
                    Active Grading Thresholds
                  </h3>
                  <span className="text-[10px] text-purple-900/85 font-semibold">Config.py Settings</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-purple-950">
                  <div className="space-y-2.5">
                    <span className="text-purple-900 font-semibold block uppercase tracking-wider text-[10px]">Semantic Classification</span>
                    <div className="flex justify-between border-b border-purple-200/40 pb-1">
                      <span>Correct threshold</span>
                      <span className="text-purple-950 font-mono font-semibold">&gt;= {thresholds.semantic_correct}</span>
                    </div>
                    <div className="flex justify-between border-b border-purple-200/40 pb-1">
                      <span>Partially correct</span>
                      <span className="text-purple-955 font-mono font-semibold text-purple-950">&gt;= {thresholds.semantic_partial}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Off-topic cap</span>
                      <span className="text-purple-955 font-mono font-semibold text-purple-950">&lt; {thresholds.semantic_off_topic}</span>
                    </div>
                  </div>

                  <div className="space-y-2.5">
                    <span className="text-purple-900 font-semibold block uppercase tracking-wider text-[10px]">Keyword Coverage</span>
                    <div className="flex justify-between border-b border-purple-200/40 pb-1">
                      <span>Correct coverage</span>
                      <span className="text-purple-950 font-mono font-semibold">&gt;= {thresholds.coverage_correct}</span>
                    </div>
                    <div className="flex justify-between border-b border-purple-200/40 pb-1">
                      <span>Good coverage</span>
                      <span className="text-purple-955 font-mono font-semibold text-purple-950">&gt;= {thresholds.coverage_good}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Missing concept trigger</span>
                      <span className="text-purple-955 font-mono font-semibold text-purple-950">&lt; {thresholds.coverage_missing}</span>
                    </div>
                  </div>

                  <div className="space-y-2.5">
                    <span className="text-purple-900 font-semibold block uppercase tracking-wider text-[10px]">Logical Contradiction NLI</span>
                    <div className="flex justify-between border-b border-purple-200/40 pb-1">
                      <span>High contradiction (Cap 40)</span>
                      <span className="text-purple-950 font-mono font-semibold">&gt;= {thresholds.contradiction_high}</span>
                    </div>
                    <div className="flex justify-between border-b border-purple-200/40 pb-1">
                      <span>Moderate (Cap 55/penalty)</span>
                      <span className="text-purple-955 font-mono font-semibold text-purple-950">&gt;= {thresholds.contradiction_moderate}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Grammar acceptability</span>
                      <span className="text-purple-955 font-mono font-semibold text-purple-950">&gt;= {thresholds.grammar_good}</span>
                    </div>
                  </div>
                </div>
              </div>

            </div>
          )}

        </main>
      </div>
    </div>
  );
}
