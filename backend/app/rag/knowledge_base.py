"""
NVIDIA H100 Vector Knowledge Base Documents.
Authoritative technical reference documents covering thermal design, cooling, power limits, and diagnostic playbooks.
"""

H100_KNOWLEDGE_DOCUMENTS = [
    {
        "id": "h100_thermal_spec_01",
        "title": "NVIDIA H100 SXM5 Thermal Specifications & Operating Limits",
        "category": "hardware_specs",
        "content": (
            "The NVIDIA H100 SXM5 GPU features a maximum thermal design power (TDP) of 700 Watts. "
            "The safe junction temperature range is 30°C to 80°C. Thermal throttling initiates at "
            "90°C (TJunction Max), at which point SM clocks automatically drop from boost (~1980 MHz) "
            "to base (~1000 MHz) to prevent permanent silicon damage. Emergency hard thermal shutdown "
            "occurs at 105°C. Efficient cooling strategies must maintain operating temperatures "
            "between 60°C and 78°C under sustained 700W Tensor core workloads."
        )
    },
    {
        "id": "h100_tensor_power_02",
        "title": "Tensor Core FP8 Workload Power Density & Thermal Spikes",
        "category": "workload_thermodynamics",
        "content": (
            "FP8 Transformer Engine matrix multiplication workloads on H100 Tensor Cores induce localized "
            "die power density exceeding 2.5 W/mm². When Tensor Core utilization exceeds 90%, thermal "
            "velocity (dT/dt) can reach +1.8°C to +2.5°C per second. Traditional bios curves react with "
            "a 10-15 second thermal inertia lag, resulting in transient overshoots above 88°C. "
            "Predictive AI cooling preemption is required to ramp coolant LPM and fan speed 5 seconds "
            "prior to Tensor spike arrival."
        )
    },
    {
        "id": "h100_cooling_lpm_03",
        "title": "Liquid Cooling & Forced Convection Heat Transfer Dynamics",
        "category": "cooling_physics",
        "content": (
            "NVIDIA H100 SXM5 cold-plate liquid cooling dissipates up to 650W via liquid flow rate (1.0 to 6.0 LPM) "
            "combined with chassis air fan speed (20% to 100%). The forced convection heat dissipation formula is: "
            "Q_out = (T_gpu - T_ambient) * (k_air_base + k_fan * fan_pct + k_liquid * pump_pct). "
            "Liquid pump modulation is 8x more effective per Watt of cooling energy than air fan speed, "
            "making pump flow rate the primary thermal control lever for datacenter liquid loops."
        )
    },
    {
        "id": "h100_power_limit_04",
        "title": "Dynamic Power Limit Throttling & TDP Constraints",
        "category": "control_policy",
        "content": (
            "The H100 power limit constraint can be set dynamically via NVML/DCGM from 300W to 700W. "
            "When ambient inlet temperature rises above 32°C or fan cooling is saturated at 100%, "
            "reducing the power limit by 10-20% (e.g., from 700W to 580W) prevents junction thermal runaway "
            "while sacrificing less than 4% of total FP8 FLOPs throughput. This is the ultimate "
            "control action when cooling capacity is capped."
        )
    },
    {
        "id": "h100_arrhenius_lifespan_05",
        "title": "Arrhenius Semiconductor Lifespan & Thermal Degradation",
        "category": "hardware_longevity",
        "content": (
            "Semiconductor electromigration and thermal degradation follow Black's equation and Arrhenius rate theory. "
            "Operating an H100 GPU continuously at 85°C reduces mean time between failures (MTBF) by 45% compared "
            "to operating at 70°C. Every sustained 10°C reduction in mean GPU die temperature approximately doubles "
            "chip lifespan. Reducing micro-thermal oscillations (mechanical fan jitter) also reduces solder-joint stress."
        )
    },
    {
        "id": "h100_hbm3_mem_06",
        "title": "HBM3 High Bandwidth Memory Thermal Thresholds",
        "category": "memory_thermodynamics",
        "content": (
            "NVIDIA H100 80GB features 5 stacks of HBM3 memory operating at up to 3.35 TB/sec bandwidth. "
            "HBM3 memory refresh rates must double when memory temperature exceeds 85°C, causing a 5-8% memory "
            "bandwidth penalty. Keeping HBM3 temperature below 80°C preserves maximum memory performance "
            "and prevents read-disturb bit errors."
        )
    },
    {
        "id": "h100_emergency_playbook_07",
        "title": "Agentic AI Thermal Emergency Mitigation Playbook",
        "category": "emergency_playbook",
        "content": (
            "If GPU temperature velocity dT/dt > 2.0°C/s or temperature exceeds 86°C: "
            "1. Instantly ramp liquid pump and fan cooling to 100% (Preemptive Ramp). "
            "2. If temperature continues rising past 88°C, drop Power Limit constraint to 550W (Dynamic Cap). "
            "3. If ambient inlet temperature exceeds 36°C, cap power limit to 500W until ambient normalizes. "
            "4. Once thermal velocity decelerates (dT/dt < 0), gradually restore power limit and optimize fan speed."
        )
    }
]
