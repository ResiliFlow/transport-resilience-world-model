from __future__ import annotations

import json
from pathlib import Path
import unittest

from engine.model_router import PRIMARY_SOURCES


ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "model_library" / "cards"


def _card(name: str) -> dict[str, object]:
    return json.loads((CARDS / name).read_text(encoding="utf-8"))


class ModelLibraryCitationTests(unittest.TestCase):
    def test_corrected_primary_metadata_is_consistent(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for folder in (ROOT / "engine", ROOT / "model_library", ROOT / "docs")
            for path in folder.rglob("*")
            if path.suffix in {".py", ".json", ".md"}
        )
        for stale in (
            "10.1016/j.cor.2007.01.019",
            "10.1007/s10898-007-9216-6",
            "Path-size measures for route-set quality",
            "Route-set quality and path-size measures",
            "Measuring transportation-system performance in disasters",
            "Disaster performance of transportation systems",
        ):
            self.assertNotIn(stale, combined)
        self.assertEqual(PRIMARY_SOURCES["nagurney"]["doi"], "10.1007/s10898-007-9198-1")
        self.assertIn("Constrained Time-Dependent K Shortest Paths", PRIMARY_SOURCES["hu_chiu"]["citation"])
        self.assertIn("Travel time resilience of roadway networks under disaster", PRIMARY_SOURCES["faturechi"]["citation"])

    def test_cards_record_exact_verified_sources_and_public_ids(self) -> None:
        self.assertIn(
            "10.1016/j.cor.2006.09.019",
            json.dumps(_card("P5_M16_interdiction_protection.json")),
        )
        self.assertIn(
            "10.1007/s10898-007-9198-1",
            json.dumps(_card("P5_M11_progressive_robustness.json")),
        )
        self.assertIn(
            "A Constrained Time-Dependent K Shortest Paths Algorithm Addressing Overlap and Travel Time Deviation",
            json.dumps(_card("P5_M23_historical_route_detour.json")),
        )
        self.assertIn(
            "Distributive justice and equity in transportation",
            json.dumps(_card("P5_M15_adaptive_v1.json")),
        )
        self.assertIn(
            "A Framework to Quantitatively Assess and Enhance the Seismic Resilience of Communities",
            json.dumps(_card("P5_M19_adaptive_v1.json")),
        )
        self.assertIn(
            "Network Robustness Index: A new method for identifying critical links and evaluating the performance of transportation networks",
            json.dumps(_card("P5_M11_progressive_robustness.json")),
        )
        dpc = _card("P5_M10_spatial_hotspots.json")
        self.assertEqual(dpc["public_id"], "TWM-GEO-HOTSPOT")
        self.assertIn("10.48550/arXiv.2606.30242", json.dumps(dpc))
        self.assertIn("finite quadrature", json.dumps(dpc).lower())
        scenario = _card("P5_M21_disruption_scenario.json")
        self.assertEqual(scenario["public_id"], "TWM-DISRUPTION-COUNTERFACTUAL")
        full_tier = scenario["execution_tiers"][-1]
        requirements = " ".join(full_tier["required_data"]).lower()
        self.assertIn("capacity on every routable link", requirements)
        self.assertNotIn("lane-capacity", requirements)
        self.assertNotIn("storage supply", requirements)


if __name__ == "__main__":
    unittest.main()
