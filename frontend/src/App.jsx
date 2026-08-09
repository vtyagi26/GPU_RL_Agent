import React, { useState, useEffect, useRef } from 'react';
import {
  ResponsiveContainer, ComposedChart, Line, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine
} from 'recharts';
import {
  Activity, Cpu, Thermometer, Zap, Droplets,
  ShieldCheck, TrendingDown, TrendingUp, Wind, Waves,
  Bot, BookOpen, MessageSquare, Send, CheckCircle2,
  AlertTriangle, Sliders, Layers, Sparkles, Server, Clock
} from 'lucide-react';
import { useTelemetry } from './hooks/useTelemetry';
import './index.css';

const POLICY_MODES = [
  { id: 'Balanced', label: 'Balanced AI', desc: 'Optimal cooling vs energy efficiency' },
  { id: 'Max_Performance', label: 'Max Performance', desc: 'Prioritize max clock throughput' },
  { id: 'Eco_Silent', label: 'Eco Silent', desc: 'Low noise & low pump power' },
  { id: 'Strict_Thermal_Cap', label: 'Strict Thermal Cap', desc: 'Dynamic TDP cap at 550W' },
];

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(3,7,18,0.97)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: '0.75rem',
      padding: '0.75rem 1rem',
      fontSize: '0.8rem',
      fontFamily: 'JetBrains Mono, monospace',
      minWidth: 180,
    }}>
      <p style={{ color: '#64748b', marginBottom: 6, fontFamily: 'Inter, sans-serif', fontSize: '0.75rem' }}>{label}</p>
      {payload.map((p, i) => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 3 }}>
          <span style={{ color: p.color }}>{p.name}</span>
          <span style={{ color: '#f1f5f9', fontWeight: 600 }}>
            {p.value != null ? `${p.value.toFixed(1)}°C` : '—'}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function App() {
  const { data, latestPayload, isConnected } = useTelemetry();
  const [activePolicy, setActivePolicy] = useState('Balanced');
  const [chatQuery, setChatQuery] = useState('');
  const [chatMessages, setChatMessages] = useState([
    {
      sender: 'assistant',
      text: 'Hello! I am your H100 Thermal RAG Copilot. Ask me anything about H100 thermal specs, liquid cooling heat transfer, or current telemetry state!',
      citations: []
    }
  ]);
  const [isQuerying, setIsQuerying] = useState(false);
  const chatEndRef = useRef(null);

  const params = latestPayload?.parameters || {};
  const agentic = latestPayload?.agentic_controller || {};
  const ragDiag = latestPayload?.rag_diagnosis || {};
  const savings = latestPayload?.savings_metrics || {};
  const thermalState = latestPayload?.thermal_state || {};

  const handlePolicyChange = async (mode) => {
    setActivePolicy(mode);
    try {
      await fetch('http://localhost:8000/api/agent/policy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
    } catch (e) {
      console.error('Failed to change policy:', e);
    }
  };

  const handleSendChat = async (e) => {
    e?.preventDefault();
    if (!chatQuery.trim() || isQuerying) return;

    const userText = chatQuery;
    setChatQuery('');
    setChatMessages((prev) => [...prev, { sender: 'user', text: userText }]);
    setIsQuerying(true);

    try {
      const res = await fetch('http://localhost:8000/api/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userText }),
      });
      const data = await res.json();

      setChatMessages((prev) => [
        ...prev,
        {
          sender: 'assistant',
          text: data.answer,
          citations: data.citations || []
        }
      ]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        {
          sender: 'assistant',
          text: 'Unable to reach backend RAG engine. Please ensure FastAPI server is running on port 8000.',
          citations: []
        }
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const currentAction = agentic.action?.action_type || 'MAINTAIN_SETTINGS';
  const getActionColor = (act) => {
    if (act === 'REDUCE_POWER_LIMIT') return '#f43f5e';
    if (act === 'INCREASE_FAN_COOLING') return '#fbbf24';
    return '#34d399';
  };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)', padding: '1.5rem 2rem', position: 'relative' }}>

      {/* Background radial glow effects */}
      <div style={{ position: 'fixed', inset: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: 0 }}>
        <div style={{ position: 'absolute', top: '-10%', left: '5%', width: 600, height: 600, borderRadius: '50%', background: 'radial-gradient(circle, rgba(16,185,129,0.06) 0%, transparent 70%)', filter: 'blur(40px)' }} />
        <div style={{ position: 'absolute', bottom: '-10%', right: '5%', width: 500, height: 500, borderRadius: '50%', background: 'radial-gradient(circle, rgba(34,211,238,0.05) 0%, transparent 70%)', filter: 'blur(40px)' }} />
      </div>

      <div style={{ maxWidth: 1440, margin: '0 auto', position: 'relative', zIndex: 1 }}>

        {/* ── HEADER ── */}
        <header style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', paddingBottom: '1.25rem', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
            <div style={{ padding: '0.65rem', borderRadius: '0.875rem', background: 'rgba(16,185,129,0.12)', border: '1px solid rgba(16,185,129,0.25)' }}>
              <Bot size={24} color="#34d399" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <h1 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#f1f5f9', letterSpacing: '-0.02em' }}>
                  NVIDIA H100 Agentic AI Thermal Controller
                </h1>
                <span style={{ fontSize: '0.65rem', fontWeight: 700, background: 'rgba(52,211,153,0.12)', color: '#34d399', padding: '0.2rem 0.55rem', borderRadius: '0.4rem', border: '1px solid rgba(52,211,153,0.30)', fontFamily: 'JetBrains Mono, monospace' }}>
                  RAG + 5-STAGE AGENT
                </span>
              </div>
              <p style={{ fontSize: '0.75rem', color: '#64748b', marginTop: 2 }}>
                NVIDIA H100 SXM5 (700W TDP) · 10 Parameter Real-Time Physics &amp; Closed-Loop Control
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '0.35rem 0.75rem', borderRadius: '999px', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
              <span className={isConnected ? 'pulse' : ''} style={{ width: 7, height: 7, borderRadius: '50%', background: isConnected ? '#34d399' : '#f43f5e', display: 'inline-block' }} />
              <span style={{ fontSize: '0.7rem', fontWeight: 700, color: isConnected ? '#34d399' : '#f43f5e', fontFamily: 'JetBrains Mono, monospace' }}>
                {isConnected ? 'H100 LIVE STREAM' : 'OFFLINE'}
              </span>
            </div>

            {/* Policy Selector Buttons */}
            <div style={{ display: 'flex', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '0.875rem', padding: '0.25rem' }}>
              {POLICY_MODES.map((m) => (
                <button
                  key={m.id}
                  className={`tier-btn${activePolicy === m.id ? ' active' : ''}`}
                  onClick={() => handlePolicyChange(m.id)}
                  title={m.desc}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* ── 10 CORE PARAMETERS MONITOR GRID ── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
          
          {/* 1. GPU Temp */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Thermometer size={13} color="#f43f5e" /> GPU Temperature
              </span>
              <span style={{ fontSize: '0.62rem', color: '#34d399', fontFamily: 'JetBrains Mono, monospace', fontWeight: 700 }}>
                ΔT: -{(savings.temp_reduction_c || 0).toFixed(1)}°C
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <span style={{ fontSize: '1.55rem', fontWeight: 800, color: '#34d399', fontFamily: 'JetBrains Mono, monospace' }}>
                {params.ai_controlled_temp_c != null ? `${params.ai_controlled_temp_c.toFixed(1)}°C` : '—'}
              </span>
              <span style={{ fontSize: '0.72rem', color: '#64748b', textDecoration: 'line-through' }}>
                {params.gpu_temp_c != null ? `${params.gpu_temp_c.toFixed(1)}°` : ''}
              </span>
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>
              Velocity (dT/dt): <span style={{ color: (thermalState.temp_velocity_dt_dt || 0) > 0.5 ? '#fbbf24' : '#34d399', fontWeight: 700 }}>{(thermalState.temp_velocity_dt_dt || 0) > 0 ? '+' : ''}{(thermalState.temp_velocity_dt_dt || 0).toFixed(2)}°C/s</span>
            </div>
          </div>

          {/* 2. GPU Power Draw */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Zap size={13} color="#fbbf24" /> GPU Power Usage
              </span>
              <span style={{ fontSize: '0.62rem', color: '#fbbf24', fontFamily: 'JetBrains Mono, monospace' }}>
                Limit: {params.power_limit_w || 700}W
              </span>
            </div>
            <div style={{ fontSize: '1.55rem', fontWeight: 800, color: '#f1f5f9', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.gpu_power_w != null ? `${params.gpu_power_w.toFixed(0)} W` : '—'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>
              Unconstrained: {params.unconstrained_power_w || 0}W
            </div>
          </div>

          {/* 3. GPU Utilization */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Cpu size={13} color="#38bdf8" /> GPU Utilization
              </span>
            </div>
            <div style={{ fontSize: '1.55rem', fontWeight: 800, color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.gpu_utilization_pct != null ? `${params.gpu_utilization_pct.toFixed(0)}%` : '—'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>
              Workload: <span style={{ color: '#94a3b8', fontWeight: 600 }}>{params.workload_intensity || 'IDLE'}</span>
            </div>
          </div>

          {/* 4. Tensor Core Utilization */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Sparkles size={13} color="#a78bfa" /> Tensor Core Util
              </span>
              <span style={{ fontSize: '0.62rem', color: '#a78bfa', fontFamily: 'JetBrains Mono, monospace' }}>FP8/FP16</span>
            </div>
            <div style={{ fontSize: '1.55rem', fontWeight: 800, color: '#a78bfa', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.tensor_core_util_pct != null ? `${params.tensor_core_util_pct.toFixed(0)}%` : '—'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>
              High power density compute
            </div>
          </div>

          {/* 5. Fan Speed / Cooling */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Wind size={13} color="#22d3ee" /> Fan / Liquid Cooling
              </span>
            </div>
            <div style={{ fontSize: '1.55rem', fontWeight: 800, color: '#22d3ee', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.fan_speed_pct != null ? `${params.fan_speed_pct.toFixed(0)}%` : '—'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>
              Stock BIOS: {params.stock_bios_fan_pct || 0}%
            </div>
          </div>

          {/* 6. GPU Clock */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>
              GPU SM Clock
            </span>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#e2e8f0', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.gpu_clock_mhz != null ? `${params.gpu_clock_mhz.toFixed(0)} MHz` : '—'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>Base 1000 / Boost 1980</div>
          </div>

          {/* 7. Memory Clock */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>
              Memory Clock (HBM3)
            </span>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#e2e8f0', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.memory_clock_mhz != null ? `${params.memory_clock_mhz.toFixed(0)} MHz` : '—'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>Base 1000 / Max 2619</div>
          </div>

          {/* 8. Memory Utilization */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>
              HBM3 Memory Util
            </span>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#e2e8f0', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.memory_utilization_pct != null ? `${params.memory_utilization_pct.toFixed(0)}%` : '—'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>80 GB HBM3 Capacity</div>
          </div>

          {/* 9. Ambient Temperature */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>
              Rack Ambient Temp
            </span>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: (params.ambient_temp_c || 24) > 30 ? '#f43f5e' : '#e2e8f0', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.ambient_temp_c != null ? `${params.ambient_temp_c.toFixed(1)}°C` : '—'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>Inlet Air / Fluid Temp</div>
          </div>

          {/* 10. Power Limit Constraint */}
          <div className="glass" style={{ padding: '1rem 1.15rem' }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>
              Power Limit Constraint
            </span>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: (params.power_limit_w || 700) < 700 ? '#f43f5e' : '#34d399', fontFamily: 'JetBrains Mono, monospace' }}>
              {params.power_limit_w != null ? `${params.power_limit_w.toFixed(0)} W` : '700 W'}
            </div>
            <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: 4 }}>Max TDP Cap (300-700W)</div>
          </div>

        </div>

        {/* ── AGENTIC AI LOOP HUD (OBSERVE -> REASON -> PLAN -> ACT -> VERIFY) ── */}
        <div className="glass" style={{ padding: '1.25rem', marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Layers size={18} color="#34d399" />
              <h2 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f1f5f9' }}>
                Agentic AI Controller Execution Loop
              </h2>
            </div>

            {/* Action Dispatched Badge */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '0.35rem 0.85rem', borderRadius: '0.6rem',
              background: `${getActionColor(currentAction)}15`,
              border: `1px solid ${getActionColor(currentAction)}40`,
              fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', fontWeight: 800,
              color: getActionColor(currentAction)
            }}>
              <CheckCircle2 size={14} />
              DISPATCHED ACTION: {currentAction}
            </div>
          </div>

          {/* 5 Stage Horizontal Pipeline */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.75rem' }}>
            
            {/* Stage 1 */}
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '0.6rem', padding: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontSize: '0.65rem', fontWeight: 800, color: '#38bdf8', letterSpacing: '0.08em', marginBottom: 4 }}>
                1. OBSERVE
              </div>
              <p style={{ fontSize: '0.7rem', color: '#94a3b8', lineHeight: 1.35 }}>
                {agentic.stage_1_observe || 'Ingesting H100 telemetry & LSTM T+5 predictions...'}
              </p>
            </div>

            {/* Stage 2 */}
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '0.6rem', padding: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontSize: '0.65rem', fontWeight: 800, color: '#fbbf24', letterSpacing: '0.08em', marginBottom: 4 }}>
                2. REASON
              </div>
              <p style={{ fontSize: '0.7rem', color: '#94a3b8', lineHeight: 1.35 }}>
                {agentic.stage_2_reason || 'Evaluating thermal headroom & throttle risks...'}
              </p>
            </div>

            {/* Stage 3 */}
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '0.6rem', padding: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontSize: '0.65rem', fontWeight: 800, color: '#a78bfa', letterSpacing: '0.08em', marginBottom: 4 }}>
                3. PLAN
              </div>
              <p style={{ fontSize: '0.7rem', color: '#94a3b8', lineHeight: 1.35 }}>
                {agentic.stage_3_plan || 'Formulating preemption & power cap strategy...'}
              </p>
            </div>

            {/* Stage 4 */}
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '0.6rem', padding: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontSize: '0.65rem', fontWeight: 800, color: getActionColor(currentAction), letterSpacing: '0.08em', marginBottom: 4 }}>
                4. ACT
              </div>
              <p style={{ fontSize: '0.7rem', color: '#94a3b8', lineHeight: 1.35 }}>
                {agentic.stage_4_act || 'Issuing control commands to H100...'}
              </p>
            </div>

            {/* Stage 5 */}
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '0.6rem', padding: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontSize: '0.65rem', fontWeight: 800, color: '#34d399', letterSpacing: '0.08em', marginBottom: 4 }}>
                5. VERIFY
              </div>
              <p style={{ fontSize: '0.7rem', color: '#94a3b8', lineHeight: 1.35 }}>
                {agentic.stage_5_verify || 'Closed-loop verification active.'}
              </p>
            </div>

          </div>
        </div>

        {/* ── MAIN CONTENT GRID: Chart + RAG Telemetry Copilot Drawer ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 420px', gap: '1.25rem', alignItems: 'start' }}>

          {/* LEFT: LSTM Chart + Savings Counter */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            
            {/* Chart */}
            <div className="glass" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f1f5f9', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Activity size={16} color="#34d399" />
                    H100 Real-Time Thermal Trajectory &amp; LSTM T+5 Forecast
                  </h3>
                  <p style={{ fontSize: '0.7rem', color: '#64748b', marginTop: 2 }}>
                    Stock BIOS Curve vs AI Agent Control · Predictor Lead Time 25s
                  </p>
                </div>
                <div style={{ display: 'flex', gap: 12, fontSize: '0.7rem' }}>
                  <span style={{ color: '#f43f5e', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#f43f5e' }} /> Stock BIOS
                  </span>
                  <span style={{ color: '#34d399', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#34d399' }} /> AI Agent
                  </span>
                  <span style={{ color: '#fbbf24', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#fbbf24' }} /> LSTM Forecast
                  </span>
                </div>
              </div>

              <div style={{ height: 360 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gradRL" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#34d399" stopOpacity={0.20} />
                        <stop offset="95%" stopColor="#34d399" stopOpacity={0.01} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="time" stroke="#334155" tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false} />
                    <YAxis stroke="#334155" tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false} unit="°" domain={[20, 100]} />
                    <Tooltip content={<ChartTooltip />} />
                    <ReferenceLine y={90} stroke="rgba(244,63,94,0.4)" strokeDasharray="4 4" label={{ value: 'TJunction Throttle (90°C)', fill: '#f43f5e', fontSize: 10, position: 'insideTopRight' }} />
                    
                    <Area type="monotone" dataKey="actualTemp" name="Stock BIOS" stroke="#f43f5e" strokeWidth={1.5} fill="transparent" dot={false} connectNulls />
                    <Area type="monotone" dataKey="rlTemp" name="AI Agent" stroke="#34d399" strokeWidth={2} fill="url(#gradRL)" dot={false} connectNulls />
                    <Line type="monotone" dataKey="predictedTemp" name="LSTM Forecast" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="4 3" dot={false} connectNulls />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Savings & Lifespan HUD */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              <div className="glass glow-emerald" style={{ padding: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#34d399', fontSize: '0.7rem', fontWeight: 700 }}>
                  <ShieldCheck size={14} /> Arrhenius Hardware Lifespan
                </div>
                <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#34d399', fontFamily: 'JetBrains Mono, monospace', marginTop: 4 }}>
                  +{(savings.lifespan_ext_pct || 0).toFixed(4)}%
                </div>
                <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: 2 }}>
                  MTBF Extended: +{(savings.lifespan_ext_hours || 0).toFixed(1)} hrs
                </div>
              </div>

              <div className="glass" style={{ padding: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#fbbf24', fontSize: '0.7rem', fontWeight: 700 }}>
                  <Zap size={14} /> Power Energy Saved
                </div>
                <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#fbbf24', fontFamily: 'JetBrains Mono, monospace', marginTop: 4 }}>
                  {(savings.total_power_saved_kj || 0).toFixed(2)} kJ
                </div>
                <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: 2 }}>
                  Step Delta: {(savings.step_power_saved_w || 0).toFixed(1)} W
                </div>
              </div>

              <div className="glass" style={{ padding: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#22d3ee', fontSize: '0.7rem', fontWeight: 700 }}>
                  <Waves size={14} /> Coolant Volume Conserved
                </div>
                <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#22d3ee', fontFamily: 'JetBrains Mono, monospace', marginTop: 4 }}>
                  {(savings.total_coolant_saved_liters || 0).toFixed(2)} L
                </div>
                <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: 2 }}>
                  Reduced pump wear
                </div>
              </div>
            </div>

          </div>

          {/* RIGHT: RAG Executive Telemetry Diagnosis & Interactive Copilot Chat */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            
            {/* Real-Time RAG Diagnostic Card */}
            <div className="glass" style={{ padding: '1.15rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 800, color: '#f1f5f9', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <BookOpen size={15} color="#34d399" /> Real-Time RAG Thermal Diagnosis
                </span>
                <span style={{
                  fontSize: '0.6rem', fontWeight: 700, padding: '0.15rem 0.5rem', borderRadius: '0.4rem',
                  background: ragDiag.severity === 'HIGH' ? 'rgba(244,63,94,0.15)' : 'rgba(52,211,153,0.15)',
                  color: ragDiag.severity === 'HIGH' ? '#f43f5e' : '#34d399',
                  border: `1px solid ${ragDiag.severity === 'HIGH' ? '#f43f5e' : '#34d399'}40`,
                  fontFamily: 'JetBrains Mono, monospace'
                }}>
                  {ragDiag.status || 'STABLE_NOMINAL'}
                </span>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '0.5rem', padding: '0.75rem', border: '1px solid rgba(255,255,255,0.06)' }}>
                <p style={{ fontSize: '0.72rem', color: '#e2e8f0', lineHeight: 1.4, marginBottom: 6 }}>
                  {ragDiag.finding || 'Analyzing live telemetry dynamics...'}
                </p>
                <p style={{ fontSize: '0.7rem', color: '#34d399', fontWeight: 600 }}>
                  {ragDiag.recommendation}
                </p>
                <div style={{ marginTop: 6, fontSize: '0.62rem', color: '#64748b', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: 4 }}>
                  Reference: {ragDiag.reference_doc} · Headroom: {ragDiag.junction_headroom}
                </div>
              </div>
            </div>

            {/* RAG Copilot Chat Window */}
            <div className="glass" style={{ padding: '1.15rem', height: 420, display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: '0.75rem', paddingBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                <MessageSquare size={15} color="#38bdf8" />
                <h3 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#f1f5f9' }}>
                  H100 RAG Copilot Chat Assistant
                </h3>
              </div>

              {/* Chat Message Stream */}
              <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10, paddingRight: 4 }}>
                {chatMessages.map((msg, idx) => (
                  <div key={idx} style={{
                    alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                    maxWidth: '88%',
                    background: msg.sender === 'user' ? 'rgba(56,189,248,0.15)' : 'rgba(255,255,255,0.04)',
                    border: `1px solid ${msg.sender === 'user' ? 'rgba(56,189,248,0.30)' : 'rgba(255,255,255,0.08)'}`,
                    borderRadius: '0.75rem',
                    padding: '0.65rem 0.85rem',
                    fontSize: '0.74rem',
                    color: '#e2e8f0',
                    lineHeight: 1.45
                  }}>
                    {msg.text}
                    {msg.citations && msg.citations.length > 0 && (
                      <div style={{ marginTop: 6, paddingTop: 4, borderTop: '1px solid rgba(255,255,255,0.06)', fontSize: '0.62rem', color: '#38bdf8' }}>
                        Citations: {msg.citations.map((c) => c.title).join(', ')}
                      </div>
                    )}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>

              {/* Input Form */}
              <form onSubmit={handleSendChat} style={{ display: 'flex', gap: 6, marginTop: '0.75rem' }}>
                <input
                  type="text"
                  placeholder="Ask RAG Copilot about thermals, H100 specs..."
                  value={chatQuery}
                  onChange={(e) => setChatQuery(e.target.value)}
                  style={{
                    flex: 1,
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.10)',
                    borderRadius: '0.6rem',
                    padding: '0.5rem 0.75rem',
                    fontSize: '0.75rem',
                    color: '#f1f5f9',
                    outline: 'none'
                  }}
                />
                <button
                  type="submit"
                  disabled={isQuerying || !chatQuery.trim()}
                  style={{
                    background: 'rgba(56,189,248,0.20)',
                    border: '1px solid rgba(56,189,248,0.40)',
                    color: '#38bdf8',
                    borderRadius: '0.6rem',
                    padding: '0 0.85rem',
                    cursor: 'pointer',
                    fontWeight: 700
                  }}
                >
                  <Send size={14} />
                </button>
              </form>

            </div>

          </div>

        </div>

        {/* Footer */}
        <footer style={{ marginTop: '2rem', textAlign: 'center', fontSize: '0.7rem', color: '#475569', paddingTop: '1.25rem', borderTop: '1px solid rgba(255,255,255,0.04)' }}>
          NVIDIA H100 Agentic AI Thermal Controller &middot; Telemetry Collector &rarr; Thermal State Engine &rarr; PyTorch LSTM &rarr; 5-Stage Agentic Controller &rarr; RAG Engine
        </footer>

      </div>
    </div>
  );
}