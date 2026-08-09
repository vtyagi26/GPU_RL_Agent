import { useState, useEffect, useRef } from 'react';

export const useTelemetry = () => {
  const [data, setData] = useState([]);
  const [latestPayload, setLatestPayload] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const historyMapRef = useRef({});

  useEffect(() => {
    setData([]);
    setLatestPayload(null);
    historyMapRef.current = {};

    const wsUrl = `ws://localhost:8000/ws/telemetry/h100`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const currentTime = payload.timestamp ? new Date(payload.timestamp) : new Date();
        const currentTs = currentTime.getTime();
        const futureTs = currentTs + 5000;

        const formatTime = (date) =>
          date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        const params = payload.parameters || {};
        const actualTemp = params.gpu_temp_c ?? payload.actual_temp ?? 35;
        const aiTemp = params.ai_controlled_temp_c ?? payload.rl_controlled_temp ?? 35;
        const predictedT5 = payload.predicted_temp_t5 ?? actualTemp;

        // Current actual & AI temp
        if (!historyMapRef.current[currentTs]) {
          historyMapRef.current[currentTs] = { ts: currentTs, time: formatTime(currentTime) };
        }
        historyMapRef.current[currentTs].actualTemp = actualTemp;
        historyMapRef.current[currentTs].rlTemp = aiTemp;
        historyMapRef.current[currentTs].power = params.gpu_power_w || 0;
        historyMapRef.current[currentTs].util = params.gpu_utilization_pct || 0;

        // Project LSTM T+5s forecast
        if (!historyMapRef.current[futureTs]) {
          historyMapRef.current[futureTs] = { ts: futureTs, time: formatTime(new Date(futureTs)) };
        }
        historyMapRef.current[futureTs].predictedTemp = predictedT5;

        setLatestPayload(payload);

        // Maintain buffer of latest 35 ticks
        const sortedKeys = Object.keys(historyMapRef.current).map(Number).sort((a, b) => a - b);
        if (sortedKeys.length > 35) {
          const keysToRemove = sortedKeys.slice(0, sortedKeys.length - 35);
          keysToRemove.forEach((k) => delete historyMapRef.current[k]);
        }

        const newChartData = sortedKeys.map((k) => historyMapRef.current[k]);
        setData(newChartData);
      } catch (err) {
        console.error("Error parsing WebSocket payload:", err);
      }
    };

    ws.onclose = () => setIsConnected(false);
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return { data, latestPayload, isConnected };
};