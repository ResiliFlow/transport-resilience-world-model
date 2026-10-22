from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ManualMatsimAndRouteIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")

    def test_route_planner_offers_only_fresh_disruption_or_historical_time(self) -> None:
        start = self.script.index("function renderRouteScenarioOptions")
        end = self.script.index("function setRoutePickRole", start)
        renderer = self.script[start:end]
        self.assertIn("Create a temporary disruption here", renderer)
        self.assertIn("Selected historical time", renderer)
        self.assertNotIn("state.scenarios.map", renderer)
        self.assertNotIn("Saved scenario", renderer)
        self.assertIn('renderRouteScenarioOptions("__quick__")', self.script)

    def test_manual_scenario_and_matsim_path_never_returns_to_copilot(self) -> None:
        self.assertIn(
            '$("#save-scenario").addEventListener("click", () => saveScenarioDraft({ manualExecution: true }))',
            self.script,
        )
        self.assertIn(
            'const pendingMatsimTask = !manualExecution && pendingAssistantTaskMatches("matsim")',
            self.script,
        )
        self.assertIn(
            'if (!manualExecution && pendingAssistantTaskMatches("scenario"))',
            self.script,
        )
        self.assertIn(
            '$("#run-matsim-scenario").addEventListener("click", () => runActiveScenarioMatsim())',
            self.script,
        )


if __name__ == "__main__":
    unittest.main()
