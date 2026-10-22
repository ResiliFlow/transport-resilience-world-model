from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from engine.importers import import_csv_project, preview_csv
from engine.network_mapping import network_features, network_mapping_status


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _import(files: list[Path], database: Path) -> dict:
    previews = [preview_csv(path) for path in files]
    mappings = {path.name: preview["suggested_mapping"]["roles"] for path, preview in zip(files, previews)}
    return import_csv_project(
        files=files,
        mapping=mappings[files[0].name],
        file_mappings=mappings,
        units={"speed": "km/h", "travel_time": "min", "primary_metric": "uploaded unit"},
        timezone="unspecified",
        topology_node_scope="global",
        entity_id_scope="global",
        database_path=database,
        project_id="structure_test",
        project_name="Structure test",
        hazard_type="flood",
        source="unit_test",
    )


class TransportStructureCompatibilityTests(unittest.TestCase):
    def test_mixed_node_link_od_scenario_csv_is_split_and_joined(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "mixed_transport_package.csv"
            fields = [
                "record_type", "node_id", "lon", "lat", "link_id", "source", "target", "geometry",
                "length", "time", "speed_kmh", "flood_exposure", "origin", "destination", "demand",
                "scenario_id", "component_type", "component_id", "availability", "time_multiplier", "capacity_multiplier",
            ]
            rows = [
                {"record_type": "node", "node_id": "N1", "lon": 150.0, "lat": -33.0},
                {"record_type": "node", "node_id": "N2", "lon": 150.1, "lat": -33.0},
                {"record_type": "node", "node_id": "N3", "lon": 150.2, "lat": -33.05},
                {"record_type": "link", "link_id": "L1", "source": "N1", "target": "N2", "geometry": "LINESTRING(150 -33,150.05 -33.01,150.1 -33)", "length": 10, "time": 12, "speed_kmh": 50, "flood_exposure": 0.4},
                {"record_type": "link", "link_id": "L2", "source": "N2", "target": "N3", "geometry": "LINESTRING(150.1 -33,150.2 -33.05)", "length": 12, "time": 14.4, "speed_kmh": 50, "flood_exposure": 0.8},
                {"record_type": "od", "origin": "N1", "destination": "N3", "demand": 120},
                {"record_type": "scenario", "scenario_id": "F1", "component_type": "link", "component_id": "L2", "availability": 0, "time_multiplier": 8, "capacity_multiplier": 0},
            ]
            _write(source, fields, rows)
            preview = preview_csv(source)
            self.assertEqual(preview["transport_structure"]["structure_type"], "mixed_transport_package")
            self.assertEqual(preview["suggested_mapping"]["roles"]["travel_time"], "time")
            self.assertIsNone(preview["suggested_mapping"]["roles"]["time"])
            summary = _import([source], root / "mixed.sqlite")
            self.assertEqual(summary["transport_structure"]["node_count"], 3)
            self.assertEqual(summary["transport_structure"]["link_count"], 2)
            self.assertEqual(summary["transport_structure"]["od_count"], 1)
            self.assertEqual(summary["transport_structure"]["scenario_effect_count"], 1)
            self.assertEqual(summary["preferred_map_view"], "links")
            collection = network_features(root / "mixed.sqlite", metric="travel_time", limit=100)
            self.assertEqual(len(collection["features"]), 2)
            self.assertEqual(collection["features"][0]["geometry"]["type"], "LineString")
            self.assertTrue(any(len(feature["geometry"]["coordinates"]) == 3 for feature in collection["features"]))

    def test_direct_endpoint_link_table_opens_as_links_without_detector_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "roads.csv"
            fields = ["edge_key", "u", "v", "u_lon", "u_lat", "v_lon", "v_lat", "speed_kmh", "travel_time_min"]
            _write(source, fields, [
                {"edge_key": "E1", "u": "A", "v": "B", "u_lon": 151.0, "u_lat": -33.8, "v_lon": 151.02, "v_lat": -33.81, "speed_kmh": 40, "travel_time_min": 4},
                {"edge_key": "E2", "u": "B", "v": "C", "u_lon": 151.02, "u_lat": -33.81, "v_lon": 151.04, "v_lat": -33.82, "speed_kmh": 30, "travel_time_min": 5},
            ])
            summary = _import([source], root / "endpoints.sqlite")
            status = network_mapping_status(root / "endpoints.sqlite")
            self.assertEqual(summary["preferred_map_view"], "links")
            self.assertTrue(status["exact_geometry"])
            self.assertEqual(status["mapped_link_count"], 2)

    def test_link_rows_with_embedded_od_fields_remain_links(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "mixed_road_attributes.csv"
            fields = [
                "road_id", "from_stop", "to_stop", "from_lon", "from_lat", "to_lon", "to_lat",
                "travel_time", "free_flow_time", "length", "capacity", "speed_limit_kph",
                "weekday_volume", "origin_zone", "destination_zone", "demand",
            ]
            _write(source, fields, [
                {"road_id": "R1", "from_stop": "A", "to_stop": "B", "from_lon": 151.20, "from_lat": -33.86, "to_lon": 151.21, "to_lat": -33.87, "travel_time": 6, "free_flow_time": 5, "length": 1.1, "capacity": 1800, "speed_limit_kph": 60, "weekday_volume": 24000, "origin_zone": "A", "destination_zone": "C", "demand": 120},
                {"road_id": "R2", "from_stop": "B", "to_stop": "C", "from_lon": 151.21, "from_lat": -33.87, "to_lon": 151.22, "to_lat": -33.88, "travel_time": 7, "free_flow_time": 6, "length": 1.2, "capacity": 1600, "speed_limit_kph": 50, "weekday_volume": 21000, "origin_zone": "B", "destination_zone": "C", "demand": 90},
            ])
            preview = preview_csv(source)
            structure = preview["transport_structure"]
            self.assertEqual(structure["record_counts"], {"link": 2})
            self.assertEqual(structure["roles"]["from_node"], "from_stop")
            self.assertEqual(structure["roles"]["to_node"], "to_stop")
            self.assertEqual(structure["joint_counts"]["embedded_od"], 2)
            self.assertIsNone(structure["roles"]["speed"])
            self.assertIsNone(preview["suggested_mapping"]["roles"]["time"])
            summary = _import([source], root / "embedded.sqlite")
            self.assertEqual(summary["transport_structure"]["link_count"], 2)
            self.assertEqual(summary["transport_structure"]["node_count"], 3)
            self.assertEqual(summary["transport_structure"]["od_count"], 2)
            self.assertEqual(summary["preferred_map_view"], "links")

    def test_geometry_cells_are_normalised_and_missing_topology_is_derived(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "fire_access_geometry.csv"
            _write(source, ["shape", "journey_time", "burn_severity"], [
                {"shape": json.dumps({"paths": [[[151.0, -33.8], [151.1, -33.81]]]}), "journey_time": 8, "burn_severity": 0.3},
                {"shape": "151.1,-33.81;151.2,-33.82", "journey_time": 10, "burn_severity": 0.8},
            ])
            preview = preview_csv(source)
            structure = preview["transport_structure"]
            self.assertTrue(structure["auto_importable"], structure)
            self.assertEqual(structure["joint_counts"]["generated_link_ids"], 2)
            self.assertIn("Derive stable topology endpoints", " ".join(structure["normalization"]["steps"]))
            summary = _import([source], root / "geometry.sqlite")
            status = network_mapping_status(root / "geometry.sqlite")
            self.assertEqual(summary["transport_structure"]["link_count"], 2)
            self.assertEqual(summary["transport_structure"]["node_count"], 3)
            self.assertEqual(status["topology_link_count"], 2)

    def test_unusable_table_reports_the_key_network_data_needed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "incident_notes.csv"
            _write(source, ["asset_name", "comment"], [
                {"asset_name": "bridge one", "comment": "inspection pending"},
            ])
            preview = preview_csv(source)
            normalisation = preview["transport_structure"]["normalization"]
            self.assertFalse(preview["transport_structure"]["auto_importable"])
            self.assertEqual(normalisation["status"], "needs_key_data")
            self.assertTrue(normalisation["fatal_missing"])
            self.assertIn("location pair", normalisation["fatal_missing"][0])

    def test_endpoint_alias_families_are_semantic_not_template_bound(self) -> None:
        variants = [
            (["section_ref", "start_point", "end_point", "start_lng", "start_lat", "end_lng", "end_lat", "duration"],
             ["S1", "A", "B", 151.0, -33.8, 151.1, -33.81, 6]),
            (["segment_uid", "upstream", "downstream", "x1", "y1", "x2", "y2", "capacity_vph"],
             ["S2", "A", "B", 151.0, -33.8, 151.1, -33.81, 900]),
            (["route_id", "origin", "destination", "source_x", "source_y", "target_x", "target_y", "road_length"],
             ["S3", "A", "B", 151.0, -33.8, 151.1, -33.81, 1200]),
            (["street_id", "from_station", "to_station", "a_lon", "a_lat", "b_lon", "b_lat", "flood_depth"],
             ["S4", "A", "B", 151.0, -33.8, 151.1, -33.81, 0.4]),
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index, (fields, values) in enumerate(variants, 1):
                source = root / f"variant_{index}.csv"
                _write(source, fields, [dict(zip(fields, values))])
                preview = preview_csv(source)
                structure = preview["transport_structure"]
                self.assertTrue(structure["auto_importable"], (fields, structure))
                self.assertEqual(structure["record_counts"], {"link": 1})
                self.assertEqual(structure["normalization"]["target_view"], "geographic_links")

    def test_chinese_point_headers_use_the_standard_point_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "洪水检测器.csv"
            _write(source, ["检测器编号", "纬度", "经度", "观测时间", "平均速度"], [
                {"检测器编号": "D1", "纬度": -27.4, "经度": 153.0, "观测时间": "2025-01-01T00:00:00Z", "平均速度": 42},
            ])
            preview = preview_csv(source)
            mapping = preview["suggested_mapping"]["roles"]
            self.assertEqual(mapping["entity_id"], "检测器编号")
            self.assertEqual(mapping["latitude"], "纬度")
            self.assertEqual(mapping["longitude"], "经度")
            self.assertEqual(mapping["speed"], "平均速度")
            self.assertEqual(preview["mapping_health"]["status"], "ready")
            summary = _import([source], root / "chinese_points.sqlite")
            self.assertEqual(summary["entity_count"], 1)

    def test_static_points_without_ids_receive_disclosed_generated_ids(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "landslide_locations.csv"
            _write(source, ["longitude", "latitude", "damage"], [
                {"longitude": 146.8, "latitude": -41.4, "damage": 0.7},
                {"longitude": 146.9, "latitude": -41.45, "damage": 0.4},
            ])
            preview = preview_csv(source)
            structure = preview["transport_structure"]
            self.assertTrue(structure["auto_importable"], structure)
            self.assertEqual(structure["joint_counts"]["generated_node_ids"], 2)
            self.assertTrue(structure["normalization"]["warnings"])
            summary = _import([source], root / "generated_points.sqlite")
            self.assertEqual(summary["entity_count"], 2)
            self.assertEqual(summary["preferred_map_view"], "points")

    def test_repeated_points_without_ids_keep_one_coordinate_based_history(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "unnamed_flood_sensor_history.csv"
            _write(source, ["x_coordinate", "y_coordinate", "observed_at", "water_depth"], [
                {"x_coordinate": 151.21, "y_coordinate": -33.87, "observed_at": "2025-01-01T00:00:00Z", "water_depth": 0.2},
                {"x_coordinate": 151.21, "y_coordinate": -33.87, "observed_at": "2025-01-01T00:10:00Z", "water_depth": 0.8},
            ])
            preview = preview_csv(source)
            self.assertEqual(preview["transport_structure"]["joint_counts"]["generated_node_ids"], 2)
            summary = _import([source], root / "generated_history.sqlite")
            self.assertEqual(summary["entity_count"], 1)
            self.assertEqual(summary["observation_count"], 2)
            self.assertEqual(summary["time_value_count"], 2)

    def test_separate_node_and_link_files_are_joined_by_topology(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nodes, links = root / "nodes.csv", root / "links.csv"
            _write(nodes, ["id", "x", "y"], [
                {"id": "1", "x": 151.0, "y": -33.8}, {"id": "2", "x": 151.1, "y": -33.82}, {"id": "3", "x": 151.2, "y": -33.83},
            ])
            _write(links, ["id", "from", "to", "length_km", "capacity"], [
                {"id": "R1", "from": "1", "to": "2", "length_km": 10, "capacity": 1200},
                {"id": "R2", "from": "2", "to": "3", "length_km": 11, "capacity": 1100},
            ])
            node_preview = preview_csv(nodes)
            self.assertTrue(node_preview["transport_structure"]["adaptive"], node_preview["transport_structure"])
            summary = _import([nodes, links], root / "separate.sqlite")
            self.assertEqual(summary["transport_structure"]["link_count"], 2)
            self.assertTrue(summary["transport_structure"]["geographic"])
            self.assertEqual(len(network_features(root / "separate.sqlite", metric="capacity", limit=100)["features"]), 2)

    def test_topology_only_edge_list_gets_disclosed_non_geographic_layout(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "edges.csv"
            _write(source, ["id", "from", "to", "capacity"], [
                {"id": "e1", "from": "a", "to": "b", "capacity": 100},
                {"id": "e2", "from": "b", "to": "c", "capacity": 90},
                {"id": "e3", "from": "c", "to": "a", "capacity": 80},
            ])
            summary = _import([source], root / "topology.sqlite")
            self.assertFalse(summary["transport_structure"]["geographic"])
            self.assertEqual(summary["geometry_kind"], "topology_layout_line")
            self.assertEqual(summary["capabilities"]["spatial_hotspots"], "unavailable")
            self.assertEqual(len(network_features(root / "topology.sqlite", metric="capacity", limit=100)["features"]), 3)

    def test_geojson_link_geometry_needs_no_nodes_or_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "wildfire_roads.csv"
            geometry = json.dumps({"type": "LineString", "coordinates": [[144.9, -37.8], [145.0, -37.82], [145.1, -37.84]]})
            _write(source, ["road_id", "geojson", "surface_serviceability"], [
                {"road_id": "fire-road-1", "geojson": geometry, "surface_serviceability": 0.7}
            ])
            preview = preview_csv(source)
            self.assertEqual(preview["transport_structure"]["roles"]["primary_metric"], "surface_serviceability")
            summary = _import([source], root / "geojson.sqlite")
            self.assertEqual(summary["transport_structure"]["link_count"], 1)
            self.assertEqual(summary["preferred_map_view"], "links")
            feature = network_features(root / "geojson.sqlite", metric="primary", limit=100)["features"][0]
            self.assertEqual(len(feature["geometry"]["coordinates"]), 3)

    def test_geometry_only_node_file_is_viewable_without_inventing_a_metric(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "nodes.csv"
            _write(source, ["id", "longitude", "latitude"], [
                {"id": "n1", "longitude": 130.8, "latitude": -12.4},
                {"id": "n2", "longitude": 130.9, "latitude": -12.45},
            ])
            preview = preview_csv(source)
            self.assertTrue(preview["transport_structure"]["adaptive"], preview["transport_structure"])
            summary = _import([source], root / "nodes.sqlite")
            self.assertEqual(summary["entity_count"], 2)
            self.assertTrue(summary["quality"]["geometry_only"])
            self.assertEqual(summary["preferred_map_view"], "points")

    def test_point_geometry_links_od_matrix_and_closures_form_one_package(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nodes = root / "junction_inventory.csv"
            links = root / "road_sections.csv"
            demand = root / "evacuation_demand_matrix.csv"
            closures = root / "wildfire_closures.csv"
            _write(nodes, ["FID", "the_geom"], [
                {"FID": "J1", "the_geom": "POINT(149.10 -35.30)"},
                {"FID": "J2", "the_geom": "POINT(149.12 -35.31)"},
                {"FID": "J3", "the_geom": "POINT(149.15 -35.32)"},
            ])
            _write(links, ["section_ref", "a_node", "b_node", "vehicles_per_hour", "burn_severity"], [
                {"section_ref": "S1", "a_node": "J1", "b_node": "J2", "vehicles_per_hour": 500, "burn_severity": 0.2},
                {"section_ref": "S2", "a_node": "J2", "b_node": "J3", "vehicles_per_hour": 350, "burn_severity": 0.8},
            ])
            _write(demand, ["origin_zone", "J1", "J2", "J3"], [
                {"origin_zone": "J1", "J1": 0, "J2": 20, "J3": 80},
                {"origin_zone": "J2", "J1": 15, "J2": 0, "J3": 30},
                {"origin_zone": "J3", "J1": 60, "J2": 25, "J3": 0},
            ])
            _write(closures, ["road_id", "closed", "delay_factor"], [
                {"road_id": "S2", "closed": "yes", "delay_factor": 4},
            ])

            summary = _import([nodes, links, demand, closures], root / "package.sqlite")
            structure = summary["transport_structure"]
            self.assertEqual(structure["node_count"], 3)
            self.assertEqual(structure["link_count"], 2)
            self.assertEqual(structure["od_count"], 6)
            self.assertEqual(structure["scenario_effect_count"], 1)
            self.assertEqual(summary["preferred_map_view"], "links")
            connection = sqlite3.connect(root / "package.sqlite")
            try:
                availability, scenario_id = connection.execute(
                    "SELECT availability, scenario_id FROM transport_scenario_effects"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(availability, 0.0)
            self.assertEqual(scenario_id, "wildfire_closures")

    def test_web_mercator_endpoint_links_are_converted_and_open_directly(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "flooded_streets.csv"
            _write(source, ["object_id", "start_id", "end_id", "x1", "y1", "x2", "y2", "water_depth"], [
                {"object_id": "W1", "start_id": "A", "end_id": "B", "x1": 16809243.1, "y1": -4015382.4, "x2": 16810356.3, "y2": -4016734.2, "water_depth": 0.6},
                {"object_id": "W2", "start_id": "B", "end_id": "C", "x1": 16810356.3, "y1": -4016734.2, "x2": 16811469.5, "y2": -4018086.3, "water_depth": 1.2},
            ])
            preview = preview_csv(source)
            self.assertEqual(preview["transport_structure"]["structure_type"], "direct_link_geometry")
            summary = _import([source], root / "mercator.sqlite")
            self.assertEqual(summary["transport_structure"]["geography_mode"], "web_mercator_to_wgs84")
            self.assertTrue(summary["transport_structure"]["geographic"])
            self.assertEqual(summary["preferred_map_view"], "links")

    def test_unknown_local_grid_is_visualised_without_false_geography(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "earthquake_local_grid.csv"
            _write(source, ["road_id", "from", "to", "from_easting", "from_northing", "to_easting", "to_northing", "damage"], [
                {"road_id": "U1", "from": "A", "to": "B", "from_easting": 334000, "from_northing": 6251000, "to_easting": 334500, "to_northing": 6251500, "damage": 0.6},
                {"road_id": "U2", "from": "B", "to": "C", "from_easting": 334500, "from_northing": 6251500, "to_easting": 335000, "to_northing": 6251800, "damage": 0.3},
            ])
            preview = preview_csv(source)
            self.assertEqual(preview["transport_structure"]["coordinate_mode"], "normalised_uploaded_geometry")
            summary = _import([source], root / "local.sqlite")
            self.assertFalse(summary["transport_structure"]["geographic"])
            self.assertEqual(summary["transport_structure"]["geography_mode"], "normalised_uploaded_geometry")
            self.assertEqual(summary["preferred_map_view"], "links")

    def test_adjacency_matrix_gets_a_disclosed_topology_view(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "earthquake_adjacency_matrix.csv"
            _write(source, ["node_id", "A", "B", "C"], [
                {"node_id": "A", "A": 0, "B": 1, "C": 0},
                {"node_id": "B", "A": 1, "B": 0, "C": 2.5},
                {"node_id": "C", "A": 0, "B": 2.5, "C": 0},
            ])
            preview = preview_csv(source)
            self.assertEqual(preview["transport_structure"]["structure_type"], "topology_matrix")
            summary = _import([source], root / "matrix.sqlite")
            self.assertEqual(summary["transport_structure"]["link_count"], 4)
            self.assertFalse(summary["transport_structure"]["geographic"])
            self.assertEqual(summary["preferred_map_view"], "links")

    def test_flat_detector_observations_stay_on_the_existing_point_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "detectors.csv"
            _write(source, ["detector_id", "lat", "lon", "datetime", "speed_kmh"], [
                {"detector_id": "D1", "lat": -27.4, "lon": 153.0, "datetime": "2025-01-01T00:00:00Z", "speed_kmh": 45},
            ])
            preview = preview_csv(source)
            self.assertFalse(preview["transport_structure"]["adaptive"])
            self.assertEqual(preview["transport_structure"]["preferred_view"], "points")


if __name__ == "__main__":
    unittest.main()
