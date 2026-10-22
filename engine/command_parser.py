from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from engine.capabilities import enrich_project_summary
from engine.natural_language import extract_requested_count
from engine.model_router import model_recommendation


INTENT_LABELS = {
    "persistent_degradation": "Find recurring problem locations",
    "critical_components": "Find critical roads",
    "area_summary": "Geographic area summary",
    "closure_scenario": "Deterministic selected-link closure",
    "restoration_screening": "Plan recovery priorities",
    "temporal_resilience": "Track disruption and recovery",
    "spatial_hotspots": "Find critical areas",
    "progressive_robustness": "Test network resilience",
    "route_detour_analysis": "Plan routes under disruption",
    "importance_exposure": "Importance and exposure sensitivity",
    "network_accessibility": "Equal-opportunity network accessibility screen",
    "resource_restoration_model": "Resource-aware staged restoration screen",
    "reliability_screen": "Reliability and sensitivity envelope",
    "cascade_screen": "Deterministic load-transfer cascade stress test",
    "hazard_exposure": "Observed hazard-transport exposure screen",
    "equity_accessibility": "Equal-weight accessibility distribution screen",
    "protection_screen": "Greedy protection-priority screen",
    "evacuation_readiness": "Evacuation-network reachability readiness screen",
    "chronological_validation": "Chronological persistence-benchmark validation",
    "investment_screen": "Resilience investment candidate screen",
}


REGISTERED_MODEL_INTENTS = {
    "importance_exposure": "P5-M5",
    "network_accessibility": "P5-M6",
    "resource_restoration_model": "P5-M8",
    "reliability_screen": "P5-M12",
    "cascade_screen": "P5-M13",
    "hazard_exposure": "P5-M14",
    "equity_accessibility": "P5-M15",
    "protection_screen": "P5-M16",
    "evacuation_readiness": "P5-M18",
    "chronological_validation": "P5-M19",
    "investment_screen": "P5-M20",
}


CORE_INTENT_MODELS = {
    "critical_components": "P5-M2",
    "restoration_screening": "P5-M4",
    "spatial_hotspots": "P5-M10",
    "progressive_robustness": "P5-M11",
    "route_detour_analysis": "P5-M23",
}


INTENT_PHRASES: dict[str, tuple[str, ...]] = {
    "temporal_resilience": (
        "resilience curve", "recovery curve", "cumulative loss", "service over time", "observed service loss", "loss and recovery",
        "track disruption and recovery", "disruption and recovery", "disruption & recovery", "how the network recovered",
        "observed resilience", "韧性曲线", "弹性曲线", "恢复曲线", "累计损失",
    ),
    "spatial_hotspots": (
        "hotspot", "hotspots", "hot spot", "hot spots", "critical area", "critical areas", "priority area", "priority areas", "risk cluster", "risk clusters",
        "热点", "关键区域", "重点区域", "高风险区域",
    ),
    "progressive_robustness": (
        "robustness curve", "progressive removal", "progressive failure", "targeted removal", "network stress test", "stress test",
        "test network resilience", "network resilience test",
        "鲁棒性曲线", "渐进失效", "连续移除", "连续删除",
    ),
    "restoration_screening": (
        "restoration", "restore", "reopen", "repair order", "recovery order", "repair priority", "recovery priority", "recovery priorities",
        "plan recovery priorities", "prioritise recovery", "prioritize recovery",
        "恢复顺序", "恢复", "修复", "抢修", "重开",
    ),
    "closure_scenario": (
        "closure", "closed", "close", "interrupt", "interruption", "failed link", "failed links", "failure scenario", "failure scenarios",
        "中断", "关闭", "封闭", "断路", "闭路", "失效路段", "封路",
    ),
    "critical_components": (
        "critical link", "critical links", "critical road", "critical roads", "critical edge", "critical edges", "important link", "important links", "important road", "important roads", "bottleneck", "bottlenecks", "bridge link", "bridge links",
        "关键边", "关键路段", "重要路段", "瓶颈路段",
    ),
    "persistent_degradation": (
        "persistent degradation", "persistently degraded", "recurrent degradation", "repeated degradation", "longest degradation",
        "find recurring problem locations", "recurring problem locations", "recurring transport problems", "repeated problem locations",
        "slowest link", "持续退化", "长期退化", "反复退化", "持续下降", "长期下降", "持续减速", "速度下降", "速度降低", "最慢路段",
    ),
    "area_summary": (
        "area summary", "inside this area", "inside the area", "within the area", "selected area",
        "区域摘要", "选定区域", "区域内", "这个范围", "所选范围",
    ),
    "route_detour_analysis": (
        "route and detour", "route & detour", "route detour", "detour analysis", "historical route",
        "compare routes", "compare route", "route comparison", "reroute between", "path between",
        "route between", "routes between", "plan a route", "plan routes",
        "route between two points", "path between two points", "routes under scenarios", "route under scenarios",
        "scenario routes", "route recommendation", "route recommendations", "plan routes under disruption", "disruption route", "disruption routes",
        "两点路径", "两点之间路径", "路径推荐", "路线推荐", "不同情景路径", "不同场景路径",
        "情景路线", "场景路线", "绕行路线", "历史路径",
    ),
    "importance_exposure": (
        "importance and exposure", "network importance", "exposure sensitivity",
        "重要性与暴露", "重要性和暴露", "网络重要性",
    ),
    "network_accessibility": (
        "network accessibility", "accessibility screen", "accessibility conditions", "weak accessibility",
        "网络可达性", "可达性筛查", "可达性状况",
    ),
    "resource_restoration_model": (
        "resource-aware restoration", "resource constrained restoration", "repair cost and duration", "crew constrained restoration",
        "资源约束恢复", "考虑资源的修复", "修复成本和工期",
    ),
    "reliability_screen": (
        "reliability screen", "failure probability", "expected efficiency loss", "sensitivity envelope", "tail risk",
        "可靠性筛查", "失效概率", "期望损失", "敏感性包络", "尾部风险",
    ),
    "cascade_screen": (
        "cascade stress", "cascading failure", "load-transfer cascade", "cascade screen",
        "级联失效", "级联中断", "负载转移级联", "级联筛查",
    ),
    "hazard_exposure": (
        "hazard exposure", "hazard-transport exposure", "hazard related hotspot", "hazard-related geographic hotspot",
        "灾害暴露", "灾害交通暴露", "灾害相关热点",
    ),
    "equity_accessibility": (
        "equity accessibility", "accessibility equity", "access inequality", "distributional accessibility",
        "公平可达性", "可达性公平", "可达性不平等", "分配公平",
    ),
    "protection_screen": (
        "protection priority", "protection screening", "protect critical links", "hardening priority", "interdiction protection",
        "保护优先级", "保护筛查", "关键路段保护", "加固优先级", "阻断保护",
    ),
    "evacuation_readiness": (
        "evacuation readiness", "evacuation accessibility", "shelter access readiness", "evacuation reachability",
        "疏散准备度", "疏散可达性", "避难所可达性", "疏散网络准备",
    ),
    "chronological_validation": (
        "chronological validation", "holdout validation", "persistence benchmark", "validate predictions", "model validation",
        "时间留出验证", "按时间验证", "持久性基准", "预测验证", "模型验证",
    ),
    "investment_screen": (
        "resilience investment", "investment priority", "investment screening", "investment portfolio",
        "韧性投资", "弹性投资", "投资优先级", "投资筛查", "投资组合",
    ),
}

FUZZY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "temporal_resilience": ("resilience", "cumulative"),
    "spatial_hotspots": ("hotspot", "hotspots"),
    "progressive_robustness": ("robustness", "progressive"),
    "restoration_screening": ("restoration", "restore", "reopen", "repair"),
    "closure_scenario": ("closure", "interrupt", "interruption", "closed"),
    "critical_components": ("critical", "bottleneck"),
    "persistent_degradation": ("persistent", "degradation", "recurrent"),
    "route_detour_analysis": ("detour", "reroute", "routing"),
    "network_accessibility": ("accessibility",),
    "reliability_screen": ("reliability",),
    "cascade_screen": ("cascading", "cascade"),
    "protection_screen": ("protection", "hardening"),
    "chronological_validation": ("validation", "holdout"),
}


def _contains_phrase(text: str, phrase: str) -> bool:
    if re.search(r"[\u3400-\u9fff]", phrase):
        return phrase in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text))


def _intent_candidates(cleaned: str, bbox: list[float] | None) -> list[str]:
    matches = {
        intent
        for intent, phrases in INTENT_PHRASES.items()
        if any(_contains_phrase(cleaned, phrase) for phrase in phrases)
    }
    # Fuzzy matching is a typo fallback, not a second classifier layered on
    # top of an already explicit phrase. This both reduces false ambiguity and
    # keeps high-volume local interpretation fast.
    if not matches:
        latin_tokens = re.findall(r"[a-z][a-z0-9-]{3,}", cleaned)
        for intent, keywords in FUZZY_KEYWORDS.items():
            if any(
                SequenceMatcher(None, token, keyword).ratio() >= (0.80 if len(token) >= 6 and len(keyword) >= 6 else 0.84)
                for token in latin_tokens
                for keyword in keywords
            ):
                matches.add(intent)
    elif matches == {"closure_scenario"} and _contains_phrase(cleaned, "order"):
        # A misspelled restoration term can coexist with the generic phrase
        # "failed links".  Probe only this narrow context so a real closure
        # order is not silently reclassified and normal parsing stays fast.
        latin_tokens = re.findall(r"[a-z][a-z0-9-]{4,}", cleaned)
        if any(
            SequenceMatcher(None, token, keyword).ratio() >= 0.84
            for token in latin_tokens
            for keyword in ("restoration", "restore", "repair")
        ):
            matches = {"restoration_screening"}

    # Specific literature-grounded families win over generic phrases they
    # contain only for a single goal. An explicit conjunction means the user
    # really requested multiple analyses and must receive a clarification.
    multi_goal = bool(re.search(
        r"\b(?:and\s+also|as\s+well\s+as|along\s+with|together\s+with|plus|then)\b|(?:同时|以及|并且|然后|接着|随后|再)",
        cleaned,
    ))
    if not multi_goal:
        advanced_priority = (
            "equity_accessibility", "evacuation_readiness", "resource_restoration_model",
            "hazard_exposure", "chronological_validation", "protection_screen",
            "importance_exposure", "reliability_screen", "cascade_screen",
            "investment_screen", "network_accessibility",
        )
        advanced = next((intent for intent in advanced_priority if intent in matches), None)
        if advanced:
            matches = {advanced}
    if not multi_goal and "temporal_resilience" in matches and _contains_phrase(cleaned, "recovery curve"):
        matches.discard("restoration_screening")
    if not multi_goal and "spatial_hotspots" in matches and any(
        _contains_phrase(cleaned, phrase) for phrase in ("critical area", "critical areas", "priority area", "priority areas", "关键区域", "重点区域")
    ):
        matches.discard("critical_components")
    if not multi_goal and "progressive_robustness" in matches:
        matches.discard("closure_scenario")
        # The resilience stress test deliberately removes a ranked set of
        # structurally critical roads.  "critical roads" therefore describes
        # the stress-test input, not a second analysis goal.  Treating it as a
        # separate critical-road request made the assistant's own canonical
        # preview command ambiguous on its round trip through this parser.
        matches.discard("critical_components")
    if "restoration_screening" in matches and any(
        _contains_phrase(cleaned, phrase)
        for phrase in ("restoration", "restore", "repair order", "recovery order", "恢复顺序", "修复", "抢修")
    ):
        explicit_closure_goal = any(
            _contains_phrase(cleaned, phrase)
            for phrase in ("closure", "closed", "close", "interrupt", "interruption", "failure scenario", "关闭", "封闭", "封路", "断路")
        )
        if not multi_goal or not explicit_closure_goal:
            matches.discard("closure_scenario")
        if not multi_goal and any(term in cleaned for term in ("selected", "failed", "damaged", "中断", "失效", "选中")):
            matches.discard("critical_components")
    if "area_summary" in matches and len(matches) > 1 and not any(
        term in cleaned for term in ("summar", "describe", "area summary", "区域摘要", "汇总", "总结")
    ):
        matches.discard("area_summary")
    if not matches and bbox and any(term in cleaned for term in ("summar", "describe", "what is here", "看看", "汇总", "总结")):
        matches.add("area_summary")
    return sorted(matches)


def _requested_top_k(text: str, default: int = 10) -> int:
    return extract_requested_count(text, default=default)


def _clarification_contract(
    *,
    text: str,
    project: dict[str, Any],
    metric: str,
    bbox: list[float] | None,
    top_k: int,
    message: str,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ready": False,
        "status": "needs_input",
        "analysis_type": None,
        "model_id": None,
        "analysis_label": "Clarify the analysis request",
        "metric": metric,
        "threshold": 0.2,
        "top_k": top_k,
        "selected_keys": [],
        "bbox": None,
        "time_start": None,
        "time_end": None,
        "question_understood": text.strip(),
        "grounding": {
            "project_id": project["project_id"],
            "project_name": project["name"],
            "selected_link_count": 0,
            "geographic_area_selected": bool(bbox),
            "scope_label": "Not decided",
            "current_time": None,
        },
        "assumptions": ["No model will run until one unambiguous intent is confirmed."],
        "warnings": [],
        "missing": [message],
        "capability": "not_checked",
        "possible_intents": candidates or [],
        "llm_used": False,
    }


def interpret_command(
    *,
    text: str,
    project: dict[str, Any],
    selected_keys: list[str] | None = None,
    bbox: list[float] | None = None,
    current_metric: str = "speed",
    current_time: int | None = None,
) -> dict[str, Any]:
    """Bounded, deterministic intent interpretation.

    The interpreter never invents a network entity or runs a model. It only
    creates a reviewable contract grounded in the current map selection.
    """

    selected_keys = selected_keys or []
    cleaned = " ".join(text.strip().lower().split())
    if not cleaned:
        raise ValueError("Enter one clear question about the uploaded data.")

    metric = current_metric
    metric_matches = []
    if any(term in cleaned for term in ("travel time", "journey time", "delay", "旅行时间", "延误", "通行时间")):
        metric_matches.append("travel_time")
    if any(term in cleaned for term in ("speed", "velocity", "速度", "车速")):
        metric_matches.append("speed")
    if any(term in cleaned for term in ("uploaded metric", "hazard value", "flood depth", "fire intensity", "primary metric", "灾害指标", "洪水深度", "火灾强度")):
        metric_matches.append("primary")
    if len(set(metric_matches)) == 1:
        metric = metric_matches[0]

    top_k = _requested_top_k(cleaned)
    candidates = _intent_candidates(cleaned, bbox)
    negated_request = bool(re.search(
        r"\b(?:do\s+not|don't|dont|never|no\s+need\s+to)\b|(?:不要|别|无需|不需要)",
        cleaned,
    ))
    if negated_request and candidates:
        return _clarification_contract(
            text=text, project=project, metric=metric, bbox=bbox, top_k=top_k,
            message="The request contains a negation. State the one analysis you do want; no model has been selected or run.",
            candidates=candidates,
        )
    if len(set(metric_matches)) > 1:
        return _clarification_contract(
            text=text, project=project, metric=metric, bbox=bbox, top_k=top_k,
            message="More than one measure was requested. Choose speed, travel time or the uploaded metric for this run.",
            candidates=candidates,
        )
    if len(candidates) > 1:
        return _clarification_contract(
            text=text, project=project, metric=metric, bbox=bbox, top_k=top_k,
            message="The request contains more than one analysis goal. Choose one first; the others can be run as separate saved analyses.",
            candidates=candidates,
        )
    if not candidates:
        return _clarification_contract(
            text=text, project=project, metric=metric, bbox=bbox, top_k=top_k,
            message="I could not safely identify one analysis. Ask to find critical roads, find critical areas, plan recovery priorities, plan routes under disruption, test network resilience, build a disruption scenario or read uploaded history.",
        )
    intent = candidates[0]
    if intent == "route_detour_analysis":
        top_k = _requested_top_k(cleaned, default=2)
    model_id = REGISTERED_MODEL_INTENTS.get(intent) or CORE_INTENT_MODELS.get(intent)
    analysis_type = "adaptive_registered_model" if intent in REGISTERED_MODEL_INTENTS else intent
    percentage_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", cleaned)
    threshold = min(2.0, max(0.01, float(percentage_match.group(1)) / 100)) if percentage_match else 0.2

    full_scope_requested = any(
        term in cleaned
        for term in (
            "full network", "whole network", "entire network", "all links", "whole project",
            "my data", "uploaded data", "current project",
            "全网", "整个网络", "全部道路", "整个项目",
        )
    )
    area_scope_requested = any(
        term in cleaned
        for term in ("selected area", "inside the area", "within the area", "区域内", "选定区域", "所选区域")
    )
    selection_analysis = intent in {
        "closure_scenario", "restoration_screening", "resource_restoration_model", "investment_screen", "route_detour_analysis"
    }
    effective_selected_keys = selected_keys if selection_analysis or intent == "cascade_screen" else []
    use_existing_area = intent not in {
        "closure_scenario", "restoration_screening", "resource_restoration_model", "investment_screen", "route_detour_analysis"
    } or area_scope_requested
    effective_bbox = None if full_scope_requested or not use_existing_area else bbox

    capabilities = enrich_project_summary(project.get("summary"))["capabilities"]
    capability_key = ({
        "persistent_degradation": "persistent_degradation",
        "critical_components": "topology_criticality",
        "area_summary": "historical_reconstruction",
        "closure_scenario": "closure_scenario",
        "restoration_screening": "restoration_screening",
        "temporal_resilience": "temporal_resilience",
        "spatial_hotspots": "spatial_hotspots",
        "progressive_robustness": "progressive_robustness",
        "route_detour_analysis": "route_detour_analysis",
    }.get(intent) or {
        "P5-M5": "topology_criticality",
        "P5-M6": "accessibility",
        "P5-M8": "restoration_screening",
        "P5-M12": "topology_criticality",
        "P5-M13": "topology_criticality",
        "P5-M14": "spatial_hotspots",
        "P5-M15": "accessibility",
        "P5-M16": "topology_criticality",
        "P5-M18": "accessibility",
        "P5-M19": "temporal_resilience",
        "P5-M20": "restoration_screening",
    }[model_id])
    capability = capabilities.get(capability_key, "unavailable")
    missing: list[str] = []
    warnings: list[str] = []

    if selection_analysis and not effective_selected_keys:
        missing.append("Select one or more mapped roads first, or open the dedicated recovery or route tool.")
    if intent == "route_detour_analysis" and len(effective_selected_keys) != 2:
        missing.append("Choose exactly two routable map points or mapped road links: one start and one destination.")
    if intent == "area_summary" and not effective_bbox:
        missing.append("Draw and confirm a geographic area first.")
    if capability == "unavailable":
        missing.append("The uploaded data do not support this model family.")
    metric_dependent = intent in {
        "persistent_degradation", "area_summary", "temporal_resilience", "spatial_hotspots",
        "hazard_exposure", "chronological_validation",
    }
    metric_summary = project.get("summary", {}).get("metrics", {}).get(metric, {})
    if metric_dependent and not metric_summary.get("available", False):
        missing.append(f"The current project does not contain {metric.replace('_', ' ')} observations.")
    severity_required = intent in {
        "persistent_degradation", "spatial_hotspots", "hazard_exposure", "chronological_validation",
    }
    if severity_required and metric == "primary":
        semantics = metric_summary.get("semantics", {})
        if not semantics.get("severity_enabled", False):
            missing.append(
                "The uploaded metric is display-only. Confirm whether higher or lower values mean worse and provide a positive severity scale in a new import."
            )
    if severity_required and metric not in {"speed", "travel_time", "primary"}:
        missing.append(
            f"{metric.replace('_', ' ').title()} is available for display and area summaries, but no degradation direction is defined for this measure. Choose speed, travel time or a severity-configured uploaded measure."
        )
    if intent == "temporal_resilience" and metric not in {"speed", "travel_time"}:
        missing.append("The observed resilience curve currently requires speed or travel-time observations with a positive reference.")
    if capability == "conditional":
        warnings.append("The result will remain conditional because one or more operational inputs are unavailable.")
    if intent == "critical_components":
        warnings.append("This is topology-based criticality, not demand-weighted social impact.")
    if intent == "restoration_screening":
        warnings.append("The result is an adaptive marginal recovery sequence or a disclosed large-candidate structural screen, not a globally optimal crew schedule.")
    if intent == "temporal_resilience":
        warnings.append("The curve is relative to the internal reference window, not a causal no-disaster counterfactual.")
    if intent == "spatial_hotspots":
        warnings.append("Grid cells are geographic summaries, not administrative or hazard boundaries.")
    if intent == "progressive_robustness":
        warnings.append("The targeted-removal curve is deterministic and does not represent failure probability.")
    if intent == "route_detour_analysis":
        warnings.append("This is retrospective least-impedance routing on uploaded history, not live navigation or an evacuation instruction.")
    if intent == "importance_exposure":
        warnings.append("Without an OD matrix this runs the equal-demand importance tier and withholds demand exposure claims.")
    if intent in {"network_accessibility", "equity_accessibility", "evacuation_readiness"}:
        warnings.append("The executable tier is structural reachability; facilities, population groups or evacuation demand are not invented.")
    if intent in {"resource_restoration_model", "investment_screen"}:
        warnings.append("This is a marginal-benefit screening sequence, not a globally optimal crew schedule or investment portfolio.")
    if intent == "reliability_screen":
        warnings.append("Probability-weighted results run only where valid uploaded failure probabilities are available; otherwise the deterministic sensitivity tier is used.")
    if intent == "cascade_screen":
        warnings.append("The base tier uses deterministic edge-betweenness proxy load, not calibrated traffic flow or interdependent-infrastructure data.")
    if intent == "hazard_exposure":
        warnings.append("Uploaded observations are mapped as exposure evidence; no hazard-impact function is invented.")
    if intent == "protection_screen":
        warnings.append("This is greedy one-link sensitivity, not a defender-attacker-defender optimum.")
    if intent == "chronological_validation":
        warnings.append("The final 30% of each eligible history is held out chronologically to test a persistence benchmark.")
    if selected_keys and not selection_analysis and intent != "cascade_screen":
        warnings.append("The previous point selection is not used by this analysis; the latest question controls the target.")
    if bbox and full_scope_requested:
        warnings.append("The previous drawn area is ignored because the latest question requests the whole project.")

    if effective_selected_keys:
        scope_label = f"{len(effective_selected_keys)} selected link{'s' if len(effective_selected_keys) != 1 else ''}"
    elif effective_bbox:
        scope_label = "Current drawn area"
    else:
        scope_label = "Whole project"

    ready = not missing
    model_route = model_recommendation(model_id, project.get("summary")) if model_id else None
    measure_label = (
        "Best available network impedance"
        if intent in {"critical_components", "restoration_screening", "progressive_robustness"}
        else "Historical link impedance"
        if intent == "route_detour_analysis"
        else metric.replace("_", " ").title()
    )
    return {
        "ready": ready,
        "status": "ready" if ready and capability == "available" else "conditional" if ready else "needs_input",
        "analysis_type": analysis_type,
        "analysis_label": INTENT_LABELS[intent],
        "model_id": model_id,
        "model_readiness": (
            "full_model_ready"
            if model_route and model_route.get("status") == "ready"
            else "ready_with_available_data"
            if model_route and model_route.get("executable")
            else "more_data_needed"
            if model_route
            else None
        ),
        "selected_model_tier": model_route.get("selected_tier") if model_route else None,
        "highest_model_tier": model_route.get("high_fidelity_tier") if model_route else None,
        "missing_for_highest_tier": model_route.get("missing_for_high_fidelity") if model_route else [],
        "metric": metric,
        "measure_label": measure_label,
        "threshold": threshold,
        "top_k": top_k,
        "selected_keys": effective_selected_keys,
        "bbox": effective_bbox,
        "time_start": None,
        "time_end": current_time,
        "question_understood": text.strip(),
        "grounding": {
            "project_id": project["project_id"],
            "project_name": project["name"],
            "selected_link_count": len(effective_selected_keys),
            "geographic_area_selected": bool(effective_bbox),
            "scope_label": scope_label,
            "current_time": current_time,
        },
        "assumptions": [
            project.get("summary", {}).get("baseline", {}).get("warning", "Uploaded baseline assumptions apply."),
            "Numerical output is deterministic and no random disaster outcome is generated.",
        ],
        "warnings": warnings,
        "missing": missing,
        "capability": capability,
        "llm_used": False,
        "analysis_assumptions": {
            "origin_key": effective_selected_keys[0] if intent == "route_detour_analysis" and len(effective_selected_keys) == 2 else None,
            "destination_key": effective_selected_keys[1] if intent == "route_detour_analysis" and len(effective_selected_keys) == 2 else None,
            "alternative_count": top_k if intent == "route_detour_analysis" else None,
            "maximum_overlap": 0.60 if intent == "route_detour_analysis" else None,
        },
    }
