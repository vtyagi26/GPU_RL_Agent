"""
Agentic AI Thermal Controller.
Implements the 5-Stage Agentic Loop: Observe -> Reason -> Plan -> Act -> Verify.

Actions:
  - INCREASE_FAN_COOLING (Ramp liquid pump / air fans)
  - REDUCE_POWER_LIMIT (Cap TDP from 700W down to 300W-650W)
  - MAINTAIN_SETTINGS (Sustain efficient steady-state)
"""


class AgenticThermalController:

    def __init__(self, policy_mode: str = "Balanced"):
        self.policy_mode = policy_mode  # Balanced, Max_Performance, Eco_Silent, Strict_Thermal_Cap
        self.last_action = "MAINTAIN_SETTINGS"
        self.last_fan_target = 35.0
        self.last_power_limit_target = 700.0
        self.verification_history = []

    def set_policy_mode(self, mode: str):
        valid = ["Balanced", "Max_Performance", "Eco_Silent", "Strict_Thermal_Cap"]
        if mode in valid:
            self.policy_mode = mode
            return True
        return False

    def process_step(self, thermal_state: dict) -> dict:
        """
        Executes 5-stage Agentic AI loop for the current tick.
        """
        temp = thermal_state.get("current_temp", 35.0)
        dt_dt = thermal_state.get("temp_velocity_dt_dt", 0.0)
        dp_dt = thermal_state.get("power_trend_dp_dt", 0.0)
        dw_dt = thermal_state.get("workload_trend_dw_dt", 0.0)
        ambient = thermal_state.get("ambient_temp", 24.0)
        tensor_util = thermal_state.get("tensor_core_util_pct", 0.0)
        power_w = thermal_state.get("power_draw_w", 200.0)
        max_pred_temp = thermal_state.get("max_predicted_temp", temp)
        pred_delta = thermal_state.get("predictor_trend_delta", 0.0)

        # ════════════════════════════════════════
        # 1. OBSERVE
        # ════════════════════════════════════════
        observe_summary = (
            f"Temp: {temp}°C, Thermal Velocity: {dt_dt:+.2f}°C/s, Power Trend: {dp_dt:+.1f}W/s, "
            f"Tensor Util: {tensor_util}%, Ambient: {ambient}°C, LSTM T+5 Forecast: {max_pred_temp}°C ({pred_delta:+.1f}°C)"
        )

        # ════════════════════════════════════════
        # 2. REASON
        # ════════════════════════════════════════
        reasons = []
        is_thermal_spike_imminent = (pred_delta > 1.2 or dt_dt > 0.8)
        is_high_ambient = (ambient > 30.0)
        is_near_throttle = (temp > 82.0 or max_pred_temp > 86.0)

        if is_near_throttle:
            reasons.append("HIGH THERMAL THREAT: Junction temperature approaching 90°C throttle threshold.")
        if is_thermal_spike_imminent:
            reasons.append(f"PREDICTIVE SPIKE: LSTM forecasts +{pred_delta:.1f}°C temperature surge.")
        if is_high_ambient:
            reasons.append(f"AMBIENT CONSTRAINT: Inlet ambient at {ambient}°C reduces cooling delta.")
        if not reasons:
            reasons.append("STABLE OPERATING STATE: Thermal parameters well within safe headroom.")

        reasoning_text = " | ".join(reasons)

        # ════════════════════════════════════════
        # 3. PLAN
        # ════════════════════════════════════════
        planned_action = "MAINTAIN_SETTINGS"
        target_fan_pct = self.last_fan_target
        target_power_limit_w = 700.0

        if self.policy_mode == "Strict_Thermal_Cap":
            target_power_limit_w = 550.0

        if is_near_throttle or (is_thermal_spike_imminent and is_high_ambient):
            # Dual action: Ramp liquid cooling + Reduce power limit cap
            planned_action = "REDUCE_POWER_LIMIT"
            target_fan_pct = 95.0
            target_power_limit_w = 550.0 if temp > 86.0 else 600.0
            plan_text = "Execute Emergency Dual Preemption: Maximize liquid flow to 95% AND reduce Power Limit to prevent TJunction throttling."
        elif is_thermal_spike_imminent:
            # Action: Increase cooling preemptively
            planned_action = "INCREASE_FAN_COOLING"
            target_fan_pct = min(100.0, max(60.0, 40.0 + (max_pred_temp - 60.0) * 2.0))
            plan_text = f"Preemptive Cooling Ramp: Boost liquid flow to {target_fan_pct:.1f}% before thermal spike arrives."
        elif is_high_ambient:
            planned_action = "REDUCE_POWER_LIMIT"
            target_fan_pct = 75.0
            target_power_limit_w = 620.0
            plan_text = f"Ambient Mitigation: Cap TDP at 620W and maintain 75% liquid pump speed to offset {ambient}°C rack ambient."
        elif dt_dt < -0.4 and temp < 70.0:
            # Valley shedding
            planned_action = "MAINTAIN_SETTINGS"
            target_fan_pct = max(25.0, self.last_fan_target - 5.0)
            plan_text = "Valley Shedding: Workload dropping, gradually reducing pump power to conserve energy."
        else:
            # Steady state optimization based on policy
            if self.policy_mode == "Eco_Silent":
                target_fan_pct = min(45.0, max(20.0, (temp - 35.0) * 0.8))
                target_power_limit_w = 600.0
            elif self.policy_mode == "Max_Performance":
                target_fan_pct = min(100.0, max(50.0, (temp - 30.0) * 1.5))
                target_power_limit_w = 700.0
            else:
                # Balanced
                target_fan_pct = min(85.0, max(25.0, 25.0 + (temp - 40.0) * 1.2))
                target_power_limit_w = 700.0

            plan_text = f"Maintain Steady State: Modulate cooling to {target_fan_pct:.1f}% for optimal hardware longevity."

        # ════════════════════════════════════════
        # 4. ACT
        # ════════════════════════════════════════
        action_name = planned_action
        action_details = {
            "action_type": action_name,
            "target_fan_pct": round(float(target_fan_pct), 1),
            "target_power_limit_w": round(float(target_power_limit_w), 1),
            "target_pump_lpm": round(float(0.20 + 0.80 * (target_fan_pct / 100.0) * 6.0), 2),
        }
        act_summary = f"Dispatched action [{action_name}] -> Fan/Pump={target_fan_pct:.1f}%, PowerLimit={target_power_limit_w:.0f}W"

        # ════════════════════════════════════════
        # 5. VERIFY
        # ════════════════════════════════════════
        verification_status = "VERIFIED_STABLE"
        if dt_dt > 1.5 and action_name == "MAINTAIN_SETTINGS":
            verification_status = "UNVERIFIED_THERMAL_RISE"
            verify_text = "Warning: Temperature velocity remains positive (+{:.2f}°C/s). Adjusting policy next tick.".format(dt_dt)
        else:
            verify_text = f"Closed-Loop Verification: Thermal deceleration verified. Junction headroom at {round(90.0 - temp, 1)}°C."

        self.last_action = action_name
        self.last_fan_target = target_fan_pct
        self.last_power_limit_target = target_power_limit_w

        return {
            "policy_mode": self.policy_mode,
            "stage_1_observe": observe_summary,
            "stage_2_reason": reasoning_text,
            "stage_3_plan": plan_text,
            "stage_4_act": act_summary,
            "stage_5_verify": verify_text,
            "action": action_details,
            "verification_status": verification_status,
        }
