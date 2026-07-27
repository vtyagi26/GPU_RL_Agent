import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Zap, Droplets, Thermometer, Fan, Activity } from 'lucide-react';
import { useTelemetry } from './hooks/useTelemetry'; 

const TIERS = [
  { id: 'H100', name: 'NVIDIA H100 (Liquid Cooled)' },
  { id: 'RTX6000', name: 'NVIDIA RTX 6000 Ada' },
  { id: 'RTX4050', name: 'RTX 4050 (Mobile)' }
];

export default function App() {
  const [selectedTier, setSelectedTier] = useState(TIERS[0].id);
  const { data, latestData, isConnected } = useTelemetry(selectedTier);

  const formatDelta = (value) => {
    if (!value && value !== 0) return { text: "0.0%", color: "text-gray-400" };
    const sign = value > 0 ? "+" : "";
    const color = value > 0 ? "text-orange-400" : value < 0 ? "text-emerald-400" : "text-gray-400";
    return { text: `${sign}${value.toFixed(1)}%`, color };
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 font-sans">
      
      <div className="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Activity className="text-emerald-500" />
            GPU Thermal RL Optimizer
          </h1>
          <p className="text-slate-400 mt-1">2-Brain Architecture: LSTM Forecast + PPO Agent Control</p>
        </div>
        
        <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
          {TIERS.map(tier => (
            <button
              key={tier.id}
              onClick={() => setSelectedTier(tier.id)}
              className={`px-4 py-2 rounded-md transition-all text-sm font-medium ${
                selectedTier === tier.id 
                  ? 'bg-emerald-500 text-slate-950 shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {tier.name}
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-7xl mx-auto space-y-6">
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shadow-sm flex flex-col justify-center">
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
              <span className="text-sm font-semibold text-slate-300">
                {isConnected ? 'LIVE TELEMETRY' : 'DISCONNECTED'}
              </span>
            </div>
            <p className="text-xs text-slate-500">5-Second Physical Leaps</p>
          </div>

          <div className="bg-slate-900 p-5 rounded-xl border border-slate-800">
            <div className="flex items-center gap-2 text-slate-400 mb-2">
              <Thermometer size={18} />
              <h3 className="text-sm font-medium">Temperature</h3>
            </div>
            <div className="flex justify-between items-end">
              <div>
                <p className="text-xs text-slate-500">Unoptimized</p>
                <p className="text-xl font-bold text-slate-300">
                  {latestData ? latestData.actualTemp.toFixed(1) : '--'}°C
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-emerald-500">RL Optimized</p>
                <p className="text-2xl font-bold text-emerald-400">
                  {latestData ? latestData.rlTemp.toFixed(1) : '--'}°C
                </p>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 p-5 rounded-xl border border-slate-800">
            <div className="flex items-center gap-2 text-slate-400 mb-2">
              <Zap size={18} />
              <h3 className="text-sm font-medium">Total Power Saved</h3>
            </div>
            <p className="text-2xl font-bold text-yellow-400">
              {latestData ? latestData.metrics.total_power_saved.toFixed(0) : '--'} <span className="text-sm font-normal">Watts</span>
            </p>
          </div>

          <div className={`bg-slate-900 p-5 rounded-xl border border-slate-800 ${selectedTier !== 'H100' && 'opacity-50'}`}>
            <div className="flex items-center gap-2 text-slate-400 mb-2">
              <Droplets size={18} />
              <h3 className="text-sm font-medium">Coolant Saved</h3>
            </div>
            <p className="text-2xl font-bold text-blue-400">
              {latestData && selectedTier === 'H100' ? latestData.metrics.total_coolant_saved.toFixed(2) : '0.00'} <span className="text-sm font-normal">Liters</span>
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          <div className="lg:col-span-2 bg-slate-900 p-6 rounded-xl border border-slate-800">
            <h3 className="text-lg font-semibold mb-4 text-white">Thermal Trajectory & Forecast</h3>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis stroke="#94a3b8" domain={[30, selectedTier === 'H100' ? 115 : 95]} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                    itemStyle={{ fontSize: '14px' }}
                  />
                  <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '14px' }}/>
                  
                  <Line 
                    type="monotone" dataKey="actualTemp" name="Unoptimized (Raw)" 
                    stroke="#94a3b8" strokeWidth={2} dot={false} isAnimationActive={false}
                    connectNulls={true}
                  />
                  <Line 
                    type="monotone" dataKey="rlTemp" name="RL Optimized" 
                    stroke="#10b981" strokeWidth={3} dot={false} isAnimationActive={false}
                    connectNulls={true}
                  />
                  <Line 
                    type="monotone" dataKey="predictedTemp" name="LSTM Forecast (+5s)" 
                    stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" 
                    dot={{ r: 4, fill: '#f59e0b', strokeWidth: 0 }} isAnimationActive={false}
                    connectNulls={true}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
              <div className="flex items-center gap-2 mb-6">
                <Fan className="text-emerald-500 animate-spin-slow" size={24} />
                <h3 className="text-lg font-semibold text-white">Fan Control Policy</h3>
              </div>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg">
                  <span className="text-slate-400">Baseline (BIOS)</span>
                  <span className="font-mono text-lg text-slate-300">
                    {latestData?.controls ? latestData.controls.fan.unoptimized_pct.toFixed(1) : '--'}%
                  </span>
                </div>
                
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg border border-emerald-900">
                  <span className="text-emerald-400 font-medium">RL Agent Request</span>
                  <span className="font-mono text-lg text-emerald-400 font-bold">
                    {latestData?.controls ? latestData.controls.fan.optimized_pct.toFixed(1) : '--'}%
                  </span>
                </div>

                <div className="flex justify-between items-center pt-2">
                  <span className="text-slate-500 text-sm">Policy Delta</span>
                  <span className={`font-mono text-lg font-bold ${latestData?.controls ? formatDelta(latestData.controls.fan.delta_pct).color : ''}`}>
                    {latestData?.controls ? formatDelta(latestData.controls.fan.delta_pct).text : '--'}
                  </span>
                </div>
              </div>
            </div>

            {selectedTier === 'H100' && (
              <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
                <div className="flex items-center gap-2 mb-6">
                  <Droplets className="text-blue-400" size={24} />
                  <h3 className="text-lg font-semibold text-white">Pump Control Policy</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg">
                    <span className="text-slate-400">Baseline (BIOS)</span>
                    <span className="font-mono text-lg text-slate-300">
                      {latestData?.controls?.coolant ? latestData.controls.coolant.unoptimized_pct.toFixed(1) : '--'}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg border border-blue-900">
                    <span className="text-blue-400 font-medium">RL Agent Request</span>
                    <span className="font-mono text-lg text-blue-400 font-bold">
                      {latestData?.controls?.coolant ? latestData.controls.coolant.optimized_pct.toFixed(1) : '--'}%
                    </span>
                  </div>

                  <div className="flex justify-between items-center pt-2">
                    <span className="text-slate-500 text-sm">Policy Delta</span>
                    <span className={`font-mono text-lg font-bold ${latestData?.controls?.coolant ? formatDelta(latestData.controls.coolant.delta_pct).color : ''}`}>
                      {latestData?.controls?.coolant ? formatDelta(latestData.controls.coolant.delta_pct).text : '--'}
                    </span>
                  </div>
                </div>
              </div>
            )}
            
          </div>
        </div>
      </div>
    </div>
  );
}