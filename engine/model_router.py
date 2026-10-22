from __future__ import annotations

"""Data-aware routing across literature-grounded model families.

The router never treats a low-data proxy as the high-data formulation.  It
selects the strongest executable tier, records missing enrichments, and keeps
the same research question usable when a non-essential field is absent.
"""

from typing import Any


PRIMARY_SOURCES = {
    "wardrop": {"citation": "Wardrop (1952), Some theoretical aspects of road traffic research", "doi": "10.1680/ipeds.1952.11259"},
    "beckmann": {"citation": "Beckmann, McGuire and Winsten (1956), Studies in the Economics of Transportation", "url": "https://cowles.yale.edu/sites/default/files/2022-09/m3-all.pdf"},
    "bpr": {"citation": "U.S. Bureau of Public Roads (1964), Traffic Assignment Manual", "url": "https://rosap.ntl.bts.gov/view/dot/11795"},
    "jenelius": {"citation": "Jenelius, Petersen and Mattsson (2006), Importance and exposure in road network vulnerability analysis", "doi": "10.1016/j.tra.2005.11.003"},
    "bruneau": {"citation": "Bruneau et al. (2003), A Framework to Quantitatively Assess and Enhance the Seismic Resilience of Communities", "doi": "10.1193/1.1623497"},
    "nurre": {"citation": "Nurre et al. (2012), Restoring infrastructure systems", "doi": "10.1016/j.ejor.2012.07.010"},
    "maya_duque": {"citation": "Maya Duque, Dolinskaya and Sørensen (2016), Network repair crew scheduling and routing for emergency relief distribution problem", "doi": "10.1016/j.ejor.2015.06.026"},
    "faturechi": {"citation": "Faturechi and Miller-Hooks (2014), Travel time resilience of roadway networks under disaster", "doi": "10.1016/j.trb.2014.08.007"},
    "jenelius_area": {"citation": "Jenelius and Mattsson (2012), Road network vulnerability analysis of area-covering disruptions", "doi": "10.1016/j.tra.2012.02.003"},
    "scott": {"citation": "Scott et al. (2006), Network Robustness Index: A new method for identifying critical links and evaluating the performance of transportation networks", "doi": "10.1016/j.jtrangeo.2005.10.003"},
    "nagurney": {"citation": "Nagurney and Qiang (2008), A network efficiency measure with application to critical infrastructure networks", "doi": "10.1007/s10898-007-9198-1"},
    "yen": {"citation": "Yen (1971), Finding the K shortest loopless paths in a network", "doi": "10.1287/mnsc.17.11.712"},
    "hu_chiu": {"citation": "Hu and Chiu (2015), A Constrained Time-Dependent K Shortest Paths Algorithm Addressing Overlap and Travel Time Deviation", "doi": "10.1016/S2046-0430(16)30169-1"},
    "pregnolato": {"citation": "Pregnolato et al. (2017), The impact of flooding on road transport", "doi": "10.1016/j.trd.2017.06.020"},
    "hansen": {"citation": "Hansen (1959), How accessibility shapes land use", "doi": "10.1080/01944365908978307"},
    "buldyrev": {"citation": "Buldyrev et al. (2010), Catastrophic cascade of failures in interdependent networks", "doi": "10.1038/nature08932"},
    "motter_lai": {"citation": "Motter and Lai (2002), Cascade-based attacks on complex networks", "doi": "10.1103/PhysRevE.66.065102"},
    "wood": {"citation": "Wood (1993), Deterministic network interdiction", "doi": "10.1016/0895-7177(93)90236-R"},
    "sherali": {"citation": "Sherali, Carter and Hobeika (1991), Hurricane/flood evacuation location-allocation", "doi": "10.1016/0191-2615(91)90037-J"},
    "pereira": {"citation": "Pereira, Schwanen and Banister (2017), Distributive justice and equity in transportation", "doi": "10.1080/01441647.2016.1257660"},
    "dijkstra": {"citation": "Dijkstra (1959), A note on two problems in connexion with graphs", "doi": "10.1007/BF01386390"},
    "dpc": {
        "citation": "Xu, Niu, Dixit, Nair and Zhang (2026), Quantum percolation based dynamic propagation connectivity for critical-area identification in transport networks",
        "doi": "10.48550/arXiv.2606.30242",
        "evidence_status": "arXiv preprint; platform implementation numerically verified against its disclosed finite-quadrature contract",
    },
    "matsim": {
        "citation": "Horni, Nagel and Axhausen (eds.) (2016), The Multi-Agent Transport Simulation MATSim",
        "doi": "10.5334/baw",
        "url": "https://www.matsim.org/docs/",
    },
}


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "P5-M0": {"goal": "describe mapped conditions in an area", "minimum": "mapped", "base": "Mapped geographic summary", "high": "Metric-weighted area summary", "sources": []},
    "P5-M1": {"goal": "find persistent or recurrent degradation", "minimum": "temporal_metric", "base": "Gap-aware observed persistence", "high": "Baseline-calibrated persistence with event labels", "sources": ["bruneau"]},
    "P5-M2": {"goal": "find structurally critical roads", "minimum": "topology", "base": "Topology-efficiency and bridge screening", "high": "OD-demand weighted road criticality", "sources": ["jenelius", "nagurney"]},
    "P5-M3": {"goal": "test selected road closures", "minimum": "topology", "base": "Deterministic link-removal counterfactual", "high": "Observed-flow calibrated capacity proxy assignment", "sources": ["jenelius", "wardrop"]},
    "P5-M4": {"goal": "plan road or area recovery priorities", "minimum": "topology", "base": "Adaptive marginal-service or large-candidate structural recovery", "high": "Gain-per-cost or gain-per-duration recovery screen", "sources": ["nurre", "maya_duque", "jenelius_area", "faturechi"]},
    "P5-M5": {"goal": "measure importance and exposure", "minimum": "topology", "base": "Equal-demand importance sensitivity", "high": "Observed-OD importance and exposure", "sources": ["jenelius"]},
    "P5-M6": {"goal": "measure access to essential services", "minimum": "topology", "base": "Reachable-node accessibility proxy", "high": "Population/opportunity-weighted accessibility", "sources": ["hansen"]},
    "P5-M7": {"goal": "estimate rerouting and congestion", "minimum": "topology", "base": "Deterministic unit-OD all-or-nothing assignment", "high": "Observed-flow calibrated Beckmann/BPR equilibrium", "sources": ["wardrop", "beckmann", "bpr"]},
    "P5-M8": {"goal": "refine recovery priorities with uploaded resources", "minimum": "topology", "base": "Equal-resource adaptive recovery sequence", "high": "Gain-per-cost or gain-per-duration recovery screen", "sources": ["nurre", "maya_duque", "faturechi"]},
    "P5-M9": {"goal": "measure resilience over observed time", "minimum": "temporal_metric", "base": "Observed normalised loss integral", "high": "Event/calibrated counterfactual resilience curve", "sources": ["bruneau"]},
    "P5-M10": {"goal": "identify critical geographic areas", "minimum": "mapped_metric", "base": "Fixed-grid observed degradation hotspots", "high": "Regional Dynamic Propagation Connectivity under observed travel-time states", "sources": ["dpc", "jenelius_area"]},
    "P5-M11": {"goal": "test network resilience under cumulative critical-road failure", "minimum": "topology", "base": "Targeted-removal curve with service thresholds and first split", "high": "OD-demand weighted robustness curve", "sources": ["scott", "nagurney", "jenelius"]},
    "P5-M12": {"goal": "analyse reliability and uncertainty", "minimum": "topology", "base": "Deterministic one-at-a-time sensitivity envelope", "high": "Probability-weighted reliability and tail-risk analysis", "sources": ["jenelius"]},
    "P5-M13": {"goal": "screen cascading disruption", "minimum": "topology", "base": "Single-network proxy-load cascade stress test", "high": "Dependency-calibrated interdependent cascade", "sources": ["motter_lai", "buldyrev"]},
    "P5-M14": {"goal": "link hazards to transport exposure", "minimum": "mapped_metric", "base": "Observed disruption-as-exposure proxy", "high": "Hazard-layer depth/intensity impact function", "sources": ["pregnolato"]},
    "P5-M15": {"goal": "screen equity in accessibility loss", "minimum": "topology", "base": "Equal-weight accessibility distribution screen", "high": "Population-group distributive accessibility analysis", "sources": ["hansen", "pereira"]},
    "P5-M16": {"goal": "protect the network against targeted failure", "minimum": "topology", "base": "Deterministic interdiction sensitivity and greedy protection", "high": "Budgeted defender-attacker-defender optimisation", "sources": ["wood"]},
    "P5-M17": {"goal": "optimise disruption under traffic equilibrium", "minimum": "topology", "base": "AON rerouting-based disruption screen", "high": "Bilevel capacity-constrained user-equilibrium disruption", "sources": ["wardrop", "beckmann", "bpr", "wood"]},
    "P5-M18": {"goal": "screen evacuation and shelter access", "minimum": "topology", "base": "Network reachability readiness screen", "high": "Population-demand, shelter-capacity and multimodal evacuation assignment", "sources": ["sherali"]},
    "P5-M19": {"goal": "validate predictions against observations", "minimum": "temporal_metric", "base": "Chronological holdout and persistence benchmark", "high": "Observed-outcome event calibration and external validation", "sources": ["bruneau"]},
    "P5-M20": {"goal": "prioritise resilience investment", "minimum": "topology", "base": "Equal-cost marginal benefit candidate screen", "high": "Repair-cost and duration-annotated candidate screen", "sources": ["nurre"]},
    "P5-M21": {"goal": "build a closed, slower or longer-time disruption scenario", "minimum": "mapped", "base": "Grounded location or topology counterfactual", "intermediate": "Reproducible MATSim exploratory-probe scenario", "intermediate_minimum": "topology", "high": "Uploaded-demand MATSim baseline-versus-disruption experiment", "sources": ["jenelius", "jenelius_area", "nagurney", "wardrop", "matsim"]},
    "P5-M22": {"goal": "propagate a disruption through the transport network", "minimum": "topology", "base": "Deterministic unit-OD rerouting and spillover", "intermediate": "Reproducible MATSim exploratory-probe event propagation", "intermediate_minimum": "topology", "high": "Uploaded-demand MATSim event and link-load propagation", "sources": ["wardrop", "beckmann", "bpr", "jenelius", "matsim"]},
    "P5-M23": {"goal": "plan routes under disruption between two routable points or roads", "minimum": "topology", "base": "Primary route plus loopless overlap-screened alternatives", "high": "Historical-impedance disruption route alternatives", "sources": ["dijkstra", "yen", "hu_chiu", "jenelius"]},
}


EXECUTION_ADAPTERS: dict[str, dict[str, str]] = {
    "P5-M0": {"kind": "analysis", "id": "area_summary", "interaction": "draw_area"},
    "P5-M1": {"kind": "analysis", "id": "persistent_degradation", "interaction": "none"},
    "P5-M2": {"kind": "analysis", "id": "critical_components", "interaction": "none"},
    "P5-M3": {"kind": "analysis", "id": "closure_scenario", "interaction": "select_links"},
    "P5-M4": {"kind": "analysis", "id": "restoration_screening", "interaction": "choose_recovery_roads_or_areas"},
    "P5-M5": {"kind": "adaptive_registered_model", "id": "P5-M5", "interaction": "none"},
    "P5-M6": {"kind": "adaptive_registered_model", "id": "P5-M6", "interaction": "none"},
    "P5-M7": {"kind": "saved_disruption_scenario", "id": "P5-M22", "interaction": "select_initial_links"},
    "P5-M8": {"kind": "adaptive_registered_model", "id": "P5-M8", "interaction": "choose_recovery_roads_or_areas"},
    "P5-M9": {"kind": "analysis", "id": "temporal_resilience", "interaction": "none"},
    "P5-M10": {"kind": "analysis", "id": "spatial_hotspots", "interaction": "none"},
    "P5-M11": {"kind": "analysis", "id": "progressive_robustness", "interaction": "none"},
    "P5-M12": {"kind": "adaptive_registered_model", "id": "P5-M12", "interaction": "none"},
    "P5-M13": {"kind": "adaptive_registered_model", "id": "P5-M13", "interaction": "optional_initial_links"},
    "P5-M14": {"kind": "adaptive_registered_model", "id": "P5-M14", "interaction": "none"},
    "P5-M15": {"kind": "adaptive_registered_model", "id": "P5-M15", "interaction": "none"},
    "P5-M16": {"kind": "adaptive_registered_model", "id": "P5-M16", "interaction": "none"},
    "P5-M17": {"kind": "saved_disruption_scenario", "id": "P5-M22", "interaction": "select_initial_links"},
    "P5-M18": {"kind": "adaptive_registered_model", "id": "P5-M18", "interaction": "none"},
    "P5-M19": {"kind": "adaptive_registered_model", "id": "P5-M19", "interaction": "none"},
    "P5-M20": {"kind": "adaptive_registered_model", "id": "P5-M20", "interaction": "select_failed_links"},
    "P5-M21": {"kind": "saved_disruption_scenario", "id": "P5-M22", "interaction": "select_initial_links"},
    "P5-M22": {"kind": "saved_disruption_scenario", "id": "P5-M22", "interaction": "select_initial_links"},
    "P5-M23": {"kind": "analysis", "id": "route_detour_analysis", "interaction": "select_endpoints_and_disruption"},
}


# Authoritative product-level mapping.  The user-facing platform has six core
# decision tools, while the research registry contains additional supporting
# model families.  Keeping the mapping beside the router prevents an LLM from
# guessing which of the 24 registered families a core-tool name refers to.
CORE_TOOL_MODEL_MAP: tuple[dict[str, Any], ...] = (
    {
        "order": 1,
        "tool": "Find critical roads",
        "primary_model_id": "P5-M2",
        "supporting_model_ids": (),
    },
    {
        "order": 2,
        "tool": "Find critical areas",
        "primary_model_id": "P5-M10",
        "supporting_model_ids": (),
    },
    {
        "order": 3,
        "tool": "Plan recovery priorities",
        "primary_model_id": "P5-M4",
        "supporting_model_ids": (),
    },
    {
        "order": 4,
        "tool": "Plan routes under disruption",
        "primary_model_id": "P5-M23",
        "supporting_model_ids": (),
    },
    {
        "order": 5,
        "tool": "Test network resilience",
        "primary_model_id": "P5-M11",
        "supporting_model_ids": (),
    },
    {
        "order": 6,
        "tool": "Build a disruption scenario",
        "primary_model_id": "P5-M21",
        "supporting_model_ids": ("P5-M22",),
    },
)


def data_profile(summary: dict[str, Any] | None) -> dict[str, bool]:
    summary = summary or {}
    metrics = summary.get("metrics") or {}
    capabilities = summary.get("capabilities") or {}
    baseline = summary.get("baseline") or {}
    mapped = int(summary.get("mapped_entity_count") or 0) > 0
    topology = capabilities.get("topology_criticality") in {"available", "conditional"}
    temporal = len(summary.get("time_values") or []) > 1
    temporal_metric = temporal and any(metrics.get(name, {}).get("available") for name in ("speed", "travel_time", "primary"))
    mapped_metric = mapped and any(metrics.get(name, {}).get("available") for name in ("speed", "travel_time", "primary"))
    dpc_travel_time = bool(
        topology
        and mapped
        and temporal
        and (
            metrics.get("travel_time", {}).get("available")
            or (
                metrics.get("speed", {}).get("available")
                and metrics.get("length", {}).get("available")
            )
        )
    )
    matsim_impedance = bool(
        topology
        and metrics.get("length", {}).get("available")
        and (
            metrics.get("free_flow_time", {}).get("available")
            or metrics.get("travel_time", {}).get("available")
            or metrics.get("speed", {}).get("available")
        )
    )
    return {
        "mapped": mapped,
        "topology": topology,
        "temporal_metric": temporal_metric,
        "mapped_metric": mapped_metric,
        "dpc_travel_time": dpc_travel_time,
        "matsim_impedance": matsim_impedance,
        "flow": bool(metrics.get("flow", {}).get("available")),
        "capacity": bool(metrics.get("capacity", {}).get("available")),
        "free_flow_time": bool(metrics.get("free_flow_time", {}).get("available")),
        "repair_cost": bool(metrics.get("repair_cost", {}).get("available")),
        "repair_duration": bool(metrics.get("repair_duration", {}).get("available")),
        "failure_probability": bool(metrics.get("failure_probability", {}).get("available")),
        "od_demand": bool(summary.get("od_demand_available")),
        "opportunities": bool(summary.get("opportunity_locations_available")),
        "population": bool(summary.get("population_available")),
        "population_groups": bool(summary.get("population_groups_available")),
        "hazard_intensity": bool(summary.get("hazard_intensity_available")),
        "impact_function": bool(summary.get("impact_function_available")),
        "dependency_network": bool(summary.get("dependency_network_available")),
        "shelters": bool(summary.get("shelter_capacity_available")),
        "validation_outcome": bool(summary.get("validation_outcome_available")),
        "decision_budget": bool(summary.get("decision_budget_available")),
        # These evidence fields default to false.  An internal reference
        # window, filename or event-coincident date must never be promoted to
        # a validated baseline, event label or causal counterfactual.
        "event_labels": bool(
            summary.get("event_labels_available")
            or summary.get("event_label_available")
        ),
        "validated_baseline": bool(
            summary.get("validated_baseline_available")
            or summary.get("baseline_validated")
            or baseline.get("validated") is True
            or baseline.get("confirmed") is True
        ),
        "counterfactual_baseline": bool(
            summary.get("counterfactual_baseline_available")
            or summary.get("calibrated_counterfactual_available")
        ),
    }


def _high_fidelity_readiness(model_id: str, available: dict[str, bool]) -> tuple[bool, list[str]]:
    if model_id in {"P5-M4", "P5-M8", "P5-M20"}:
        resource_ready = bool(available.get("repair_cost") or available.get("repair_duration"))
        return resource_ready, [] if resource_ready else ["repair cost or repair duration"]
    requirements: dict[str, tuple[str, ...]] = {
        "P5-M1": ("event_labels", "validated_baseline"),
        "P5-M2": ("od_demand",), "P5-M3": ("flow", "capacity", "free_flow_time"),
        "P5-M5": ("od_demand",), "P5-M6": ("opportunities", "population"),
        "P5-M7": ("flow", "capacity", "free_flow_time"),
        "P5-M9": ("event_labels", "validated_baseline", "counterfactual_baseline"),
        "P5-M11": ("od_demand",), "P5-M12": ("failure_probability",),
        "P5-M10": ("dpc_travel_time",),
        "P5-M13": ("dependency_network",), "P5-M14": ("hazard_intensity", "impact_function"),
        "P5-M15": ("opportunities", "population_groups"), "P5-M16": ("repair_cost", "decision_budget"),
        "P5-M17": ("flow", "capacity", "free_flow_time", "od_demand", "decision_budget"),
        "P5-M18": ("od_demand", "shelters", "population"), "P5-M19": ("validation_outcome",),
        "P5-M21": ("od_demand", "capacity", "matsim_impedance"), "P5-M22": ("od_demand", "capacity", "matsim_impedance"),
        "P5-M23": ("temporal_metric",),
    }
    missing = [name for name in requirements.get(model_id, ()) if not available.get(name)]
    return not missing, missing


def model_recommendation(model_id: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    profile = MODEL_PROFILES[model_id]
    available = data_profile(summary)
    minimum = str(profile["minimum"])
    executable = bool(available.get(minimum))
    high_ready, missing_high = _high_fidelity_readiness(model_id, available)
    intermediate_minimum = profile.get("intermediate_minimum")
    intermediate_ready = bool(
        executable
        and intermediate_minimum
        and available.get(str(intermediate_minimum))
    )
    selected_tier = (
        profile["high"]
        if executable and high_ready
        else profile["intermediate"]
        if intermediate_ready and profile.get("intermediate")
        else profile["base"]
    )
    if not executable:
        selected_tier = "Not executable for this project"
    return {
        "model_id": model_id,
        "goal": profile["goal"],
        "executable": executable,
        "status": "ready" if executable and high_ready else "conditional" if executable else "needs_data",
        "selected_tier": selected_tier,
        "base_tier": profile["base"],
        "intermediate_tier": profile.get("intermediate"),
        "high_fidelity_tier": profile["high"],
        "minimum_data": minimum,
        "missing_for_high_fidelity": missing_high,
        "assumption_level": "uploaded_demand_and_supply" if high_ready else "reproducible_probe" if intermediate_ready else "disclosed_proxy" if executable else "not_run",
        "primary_sources": [PRIMARY_SOURCES[key] for key in profile.get("sources", [])],
        "execution_adapter": EXECUTION_ADAPTERS[model_id],
        "interaction_requirement": EXECUTION_ADAPTERS[model_id]["interaction"],
    }


def recommend_models(
    summary: dict[str, Any] | None,
    *,
    goal: str | None = None,
) -> dict[str, Any]:
    cleaned_goal = " ".join((goal or "").lower().split())
    recommendations = [model_recommendation(model_id, summary) for model_id in MODEL_PROFILES]
    for item in recommendations:
        goal_terms = {word for word in item["goal"].lower().split() if len(word) >= 4}
        requested_terms = set(cleaned_goal.split())
        item["goal_match_score"] = len(goal_terms & requested_terms)
    recommendations.sort(
        key=lambda item: (
            not item["executable"],
            -int(item["goal_match_score"]),
            item["status"] != "ready",
            int(item["model_id"].split("M")[-1]),
        )
    )
    executable = [item for item in recommendations if item["executable"]]
    return {
        "data_profile": data_profile(summary),
        "goal": goal,
        "recommended": executable[:5],
        "models": recommendations,
        "coverage": {
            "executable_model_count": len(executable),
            "registered_model_count": len(recommendations),
            "has_executable_path": bool(executable),
            "rule": "At least one honest executable tier is selected for every non-fatally-valid mapped project; high-fidelity claims remain data-gated.",
        },
    }


def enrich_registry(registry: dict[str, Any]) -> dict[str, Any]:
    enriched = {**registry, "adaptive_routing": True}
    models = []
    for model in registry.get("models", []):
        profile = MODEL_PROFILES.get(model.get("id"))
        if profile is None:
            models.append(model)
            continue
        models.append(
            {
                **model,
                "status": "adaptive_executable_family",
                "goal": profile["goal"],
                "execution_tiers": [
                    tier
                    for tier in (profile["base"], profile.get("intermediate"), profile["high"])
                    if tier
                ],
                "execution_adapter": EXECUTION_ADAPTERS[model["id"]],
                "literature_sources": [PRIMARY_SOURCES[key] for key in profile.get("sources", [])],
            }
        )
    enriched["models"] = models
    return enriched
