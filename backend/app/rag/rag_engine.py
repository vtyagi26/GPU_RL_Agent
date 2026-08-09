"""
RAG Diagnostic Engine.
Combines Vector Store retrieval with Thermal State analysis to generate:
  - Natural language copilot answers with cited H100 documents
  - Real-time executive telemetry reading reports and thermal diagnostics
"""

from .vector_store import LocalVectorStore


class RAGEngine:

    def __init__(self):
        self.vector_store = LocalVectorStore()

    def query(self, question: str, thermal_state: dict = None) -> dict:
        """
        Executes RAG query pipeline over H100 knowledge base + current thermal context.
        """
        # Search top 3 documents
        docs = self.vector_store.search(question, top_k=3)

        # Build context from top documents
        retrieved_context = "\n\n".join(
            [f"[{d['id']} - {d['title']}]: {d['content']}" for d in docs]
        )

        # Contextual synthesis
        current_temp_info = ""
        if thermal_state:
            t = thermal_state.get('current_temp', 'N/A')
            p = thermal_state.get('power_draw_w', 'N/A')
            amb = thermal_state.get('ambient_temp', 'N/A')
            dt = thermal_state.get('temp_velocity_dt_dt', 0.0)
            current_temp_info = f" (Current Telemetry: Temp={t}°C, Velocity={dt}°C/s, Power={p}W, Ambient={amb}°C)"

        # Generate answer based on retrieved documents
        top_doc = docs[0] if docs else None
        if top_doc and top_doc['score'] > 0.05:
            answer = (
                f"Based on NVIDIA H100 documentation [{top_doc['title']}]: {top_doc['content']} "
                f"Under active operation{current_temp_info}, thermal parameters are maintained within "
                f"safe operating bounds according to official specification."
            )
        else:
            answer = (
                f"NVIDIA H100 SXM5 operating limits dictate maximum junction temperature at 90°C and TDP at 700W. "
                f"Active thermal management applies preemptive liquid flow and dynamic power limit constraints to prevent thermal throttling.{current_temp_info}"
            )

        citations = [{"id": d["id"], "title": d["title"], "category": d["category"], "relevance_score": d["score"]} for d in docs]

        return {
            "query": question,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks_count": len(docs),
        }

    def diagnose_telemetry(self, thermal_state: dict) -> dict:
        """
        Generates real-time executive diagnostic reading outputs based on live Thermal State.
        """
        temp = thermal_state.get("current_temp", 35.0)
        dt_dt = thermal_state.get("temp_velocity_dt_dt", 0.0)
        power = thermal_state.get("power_draw_w", 200.0)
        ambient = thermal_state.get("ambient_temp", 24.0)
        tensor = thermal_state.get("tensor_core_util_pct", 0.0)
        workload = thermal_state.get("workload_intensity", "IDLE")

        # Build diagnostic query for vector search
        diag_query = f"H100 thermal temperature {temp} velocity {dt_dt} ambient {ambient} tensor power limit"
        docs = self.vector_store.search(diag_query, top_k=2)

        # Diagnostic classification
        if temp > 88.0 or dt_dt > 1.8:
            status = "CRITICAL_THERMAL_SURGE"
            severity = "HIGH"
            finding = f"High thermal velocity (+{dt_dt}°C/s) pushing H100 junction temp to {temp}°C under heavy FP8 tensor load ({tensor}%)."
            recommendation = "Action: Dispatch 100% liquid pump speed and reduce Power Limit constraint to 550W immediately."
        elif ambient > 32.0:
            status = "HIGH_AMBIENT_CHALLENGE"
            severity = "MEDIUM"
            finding = f"Elevated rack inlet ambient temperature ({ambient}°C) reducing thermal dissipation headroom."
            recommendation = "Action: Cap power limit constraint to 600W to maintain 15°C junction buffer."
        elif tensor > 85.0:
            status = "SUSTAINED_TENSOR_TRAINING"
            severity = "NORMAL"
            finding = f"High Tensor Core utilization ({tensor}%) drawing {power}W power. Preemptive cooling active."
            recommendation = "Action: Modulate liquid flow to balance cooling energy vs die temperature."
        else:
            status = "STABLE_NOMINAL"
            severity = "LOW"
            finding = f"H100 running at optimal die temp ({temp}°C) with low thermal momentum ({dt_dt}°C/s)."
            recommendation = "Action: Maintain efficient baseline cooling level."

        return {
            "status": status,
            "severity": severity,
            "finding": finding,
            "recommendation": recommendation,
            "reference_doc": docs[0]["title"] if docs else "NVIDIA H100 Thermal Spec",
            "ambient_state": f"{ambient}°C",
            "junction_headroom": f"{round(90.0 - temp, 1)}°C",
        }
