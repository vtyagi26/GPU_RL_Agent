import { useState, useEffect, useRef } from 'react';

export const useTelemetry = (tierId) => {
  const [data, setData] = useState([]);
  const [latestData, setLatestData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  
  const historyMapRef = useRef({});

  useEffect(() => {
    if (!tierId) return;
    setData([]);
    setLatestData(null);
    historyMapRef.current = {}; 
    
    const wsUrl = `ws://localhost:8000/ws/telemetry/${tierId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      
      const currentTime = new Date(payload.timestamp);
      const currentTs = currentTime.getTime();
      const futureTs = currentTs + 5000; 
      
      const formatTime = (date) => date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

      if (!historyMapRef.current[currentTs]) {
        historyMapRef.current[currentTs] = { ts: currentTs, time: formatTime(currentTime) };
      }
      historyMapRef.current[currentTs].actualTemp = payload.actual_temp;
      historyMapRef.current[currentTs].rlTemp = payload.rl_controlled_temp;

      if (!historyMapRef.current[futureTs]) {
        historyMapRef.current[futureTs] = { ts: futureTs, time: formatTime(new Date(futureTs)) };
      }
      historyMapRef.current[futureTs].predictedTemp = payload.predicted_temp_t5;

      setLatestData({
        time: formatTime(currentTime),
        actualTemp: payload.actual_temp,
        rlTemp: payload.rl_controlled_temp,
        controls: payload.controls,
        metrics: payload.metrics
      });
      
      const sortedKeys = Object.keys(historyMapRef.current).map(Number).sort((a, b) => a - b);
      
      if (sortedKeys.length > 25) {
        const keysToRemove = sortedKeys.slice(0, sortedKeys.length - 25);
        keysToRemove.forEach(k => delete historyMapRef.current[k]);
      }

      const newChartData = Object.keys(historyMapRef.current)
        .map(Number)
        .sort((a, b) => a - b)
        .map(k => historyMapRef.current[k]);
        
      setData(newChartData);
    };

    ws.onclose = () => setIsConnected(false);
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, [tierId]);

  return { data, latestData, isConnected };
};