from __future__ import annotations

import unittest

from backend.app.api.schemas import AnalysisRequest


class RecoveryRequestSchemaTests(unittest.TestCase):
    def test_critical_area_members_are_not_rejected_at_five_thousand_roads(self) -> None:
        selected_keys = [f"region::road-{index}" for index in range(6_777)]
        request = AnalysisRequest(
            project_id="project-large-area",
            analysis_type="restoration_screening",
            model_id="P5-M4",
            top_k=44,
            selected_keys=selected_keys,
            assumptions={"recovery_unit": "area"},
        )

        self.assertEqual(len(request.selected_keys), 6_777)


if __name__ == "__main__":
    unittest.main()
