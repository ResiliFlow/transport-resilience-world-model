"""Checks for a source-only checkout without private data or model weights."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.app.main import app
from infrastructure_perception.backend.app import stage6_deployment


class SourceDistributionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_main_health_and_page(self):
        self.assertEqual(self.client.get('/api/health').status_code, 200)
        self.assertEqual(self.client.get('/').status_code, 200)

    def test_missing_models_are_not_reported_as_ready(self):
        with patch.object(stage6_deployment, 'registry_path', Path('__absent_model_registry__')):
            response = self.client.get('/perception/api/health')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['status'], 'degraded')
            self.assertFalse(response.json()['ai_detection_enabled'])
            self.assertFalse(response.json()['stage7_on_demand_runtime_ready'])
            config = self.client.get('/perception/api/config')
            self.assertEqual(config.status_code, 200)
            self.assertFalse(config.json()['stage7']['inference_modes']['local']['ready'])
            self.assertEqual(self.client.get('/perception/api/stage7').status_code, 503)

    def test_bundled_map_assets_and_licence_are_available(self):
        for resource in ('maplibre-gl.js', 'maplibre-gl.css', 'LICENSE.maplibre.txt'):
            with self.subTest(resource=resource):
                self.assertEqual(self.client.get('/static/vendor/' + resource).status_code, 200)

    def test_public_readme_links_to_video_and_preprint(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / 'README.md').read_text(encoding='utf-8')
        self.assertIn('https://www.youtube.com/watch?v=3QHWsOzBRiw', readme)
        self.assertIn('https://arxiv.org/abs/2608.20709', readme)

    def test_public_method_panel_keeps_permissions_without_run_history(self):
        root = Path(__file__).resolve().parents[1]
        page = (root / 'infrastructure_perception/frontend/index.html').read_text(encoding='utf-8')
        script = (root / 'infrastructure_perception/frontend/app.js').read_text(encoding='utf-8')
        self.assertIn('id="ai-source-policies"', page)
        self.assertIn('id="run-contract-checks"', page)
        self.assertNotIn('training-record', page)
        self.assertNotIn('verified epoch', page)
        self.assertNotIn('training queue', script)
        self.assertNotIn('training unit', script)

    def test_provider_permissions_remain_restrictive(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads((root / 'infrastructure_perception/contracts/stage2_ai_contract.json').read_text(encoding='utf-8'))
        policies = {entry['id']: entry for entry in contract['source_policies']}
        restricted = policies['google_maps_core']
        self.assertEqual(restricted['training'], 'prohibited')
        self.assertEqual(restricted['automated_inference'], 'prohibited')
        self.assertTrue(restricted['official_terms_url'].startswith('https://'))
