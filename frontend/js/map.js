const EMPTY_COLLECTION = { type: "FeatureCollection", features: [] };

export class DisasterMap {
  constructor(containerId, callbacks = {}) {
    this.containerId = containerId;
    this.callbacks = callbacks;
    this.map = null;
    this.ready = false;
    this.mode = "point";
    this.displayMode = "points";
    this.areaDrag = null;
    this.areaBox = null;
    this.currentFeatures = EMPTY_COLLECTION;
    this.analysisFeatures = EMPTY_COLLECTION;
    this.scenarioFeatures = EMPTY_COLLECTION;
    this.scenarioDraftFeatures = EMPTY_COLLECTION;
    this.matsimFeatures = EMPTY_COLLECTION;
    this.selectedKeys = new Set();
    this.lastExtent = null;
  }

  async initialise() {
    if (!window.maplibregl) throw new Error("The geographic map library is unavailable.");
    this.map = new window.maplibregl.Map({
      container: this.containerId,
      center: [-81.5, 27.25],
      zoom: 5.1,
      minZoom: 2,
      maxZoom: 17,
      attributionControl: true,
      style: {
        version: 8,
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: '© <a href="https://www.openstreetmap.org/copyright/" target="_blank" rel="noopener noreferrer">OpenStreetMap contributors</a>',
          },
        },
        layers: [
          { id: "background", type: "background", paint: { "background-color": "#ead4d4" } },
          { id: "osm", type: "raster", source: "osm", paint: { "raster-opacity": 0.82, "raster-saturation": -0.28 } },
        ],
      },
    });
    this.map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), "top-right");
    // MapLibre's `load` event can wait indefinitely when the computer is
    // offline because the public basemap tiles remain unresolved.  Platform
    // data and controls do not depend on those tiles, so continue as soon as
    // the style graph exists; the neutral background becomes the disclosed
    // offline fallback while OSM tiles keep retrying independently.
    await new Promise((resolve, reject) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeout);
        this.map.off("styledata", styleReady);
        resolve();
      };
      const styleReady = () => {
        if (this.map.getLayer("background")) finish();
      };
      this.map.once("load", finish);
      this.map.on("styledata", styleReady);
      const timeout = window.setTimeout(() => {
        if (this.map.getLayer("background")) finish();
        else reject(new Error("The map style could not be initialised."));
      }, 3000);
      styleReady();
    });
    this._installSourcesAndLayers();
    this._installInteractions();
    this.ready = true;
    return this;
  }

  _installSourcesAndLayers() {
    // Detector data must appear as detectors immediately. The display request
    // is already capped and progressively filled, so clustering is unnecessary
    // and can make the first view look empty at a regional zoom.
    this.map.addSource("observations", { type: "geojson", data: EMPTY_COLLECTION, cluster: false });
    this.map.addSource("network-links", { type: "geojson", data: EMPTY_COLLECTION });
    this.map.addLayer({
      id: "network-link-casing",
      type: "line",
      source: "network-links",
      paint: {
        "line-color": "rgba(255,255,255,.88)",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 3.4, 10, 6.2, 15, 10],
        "line-opacity": .9,
      },
    });
    this.map.addLayer({
      id: "network-links",
      type: "line",
      source: "network-links",
      paint: {
        "line-color": [
          "case",
          ["!=", ["get", "quality"], "valid"], "#7d607d",
          ["interpolate", ["linear"], ["coalesce", ["get", "severity"], ["get", "display_score"], 0], 0, "#267a74", .15, "#bd842c", .4, "#df695d", .75, "#8a1f3b"],
        ],
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 1.6, 10, 3.5, 15, 7],
        "line-opacity": .88,
      },
    });
    this.map.addLayer({
      id: "observation-clusters",
      type: "circle",
      source: "observations",
      filter: ["has", "point_count"],
      paint: {
        "circle-color": ["step", ["get", "point_count"], "#6b7f86", 30, "#526971", 100, "#354f58"],
        "circle-radius": ["step", ["get", "point_count"], 16, 30, 22, 100, 29],
        "circle-stroke-width": 2,
        "circle-stroke-color": "rgba(255,255,255,.88)",
      },
    });
    this.map.addLayer({
      id: "observation-cluster-count",
      type: "symbol",
      source: "observations",
      filter: ["has", "point_count"],
      layout: {
        "text-field": ["get", "point_count_abbreviated"],
        "text-size": 12,
        "text-font": ["Open Sans Bold"],
      },
      paint: { "text-color": "#ffffff", "text-halo-color": "rgba(0,0,0,.2)", "text-halo-width": 1 },
    });
    this.map.addLayer({
      id: "observation-points",
      type: "circle",
      source: "observations",
      filter: ["!", ["has", "point_count"]],
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 3.2, 10, 5.5, 15, 8],
        "circle-color": [
          "case",
          ["!=", ["get", "quality"], "valid"], "#7d607d",
          ["interpolate", ["linear"], ["coalesce", ["get", "severity"], ["get", "display_score"], 0], 0, "#267a74", .15, "#bd842c", .4, "#df695d", .75, "#8a1f3b"],
        ],
        "circle-stroke-width": 1.2,
        "circle-stroke-color": "rgba(255,255,255,.88)",
        "circle-opacity": .88,
      },
    });
    this.map.addSource("selected", { type: "geojson", data: EMPTY_COLLECTION });
    this.map.addLayer({
      id: "selected-lines",
      type: "line",
      source: "selected",
      filter: ["==", ["geometry-type"], "LineString"],
      paint: { "line-color": "#f4c542", "line-width": 8, "line-opacity": .96 },
    });
    this.map.addLayer({
      id: "selected-points",
      type: "circle",
      source: "selected",
      filter: ["==", ["geometry-type"], "Point"],
      paint: { "circle-radius": 11, "circle-color": "rgba(0,0,0,0)", "circle-stroke-width": 3.2, "circle-stroke-color": "#f4c542" },
    });
    this.map.addSource("analysis-result", { type: "geojson", data: EMPTY_COLLECTION });
    this.map.addLayer({
      id: "route-baseline-links",
      type: "line",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["==", ["get", "route_role"], "baseline_route"]],
      paint: {
        "line-color": "#2563eb",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 3, 10, 5, 15, 7],
        "line-offset": -3.5,
        "line-opacity": .96,
      },
    });
    this.map.addLayer({
      id: "route-comparison-casing",
      type: "line",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["in", ["get", "route_role"], ["literal", ["comparison_route", "recommended_route"]]]],
      paint: {
        "line-color": "rgba(92,69,0,.88)",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 5.5, 10, 8, 15, 11],
        "line-offset": ["coalesce", ["get", "route_map_offset"], 0],
        "line-opacity": .95,
      },
    });
    this.map.addLayer({
      id: "route-comparison-links",
      type: "line",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["in", ["get", "route_role"], ["literal", ["comparison_route", "recommended_route"]]]],
      paint: {
        "line-color": ["coalesce", ["get", "route_colour"], "#f28c28"],
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 3, 10, 5, 15, 7],
        "line-offset": ["coalesce", ["get", "route_map_offset"], 0],
        "line-opacity": ["case", [">", ["coalesce", ["get", "route_shared_count"], 1], 1], .9, 1],
      },
    });
    this.map.addLayer({
      id: "route-alternative-casing",
      type: "line",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["==", ["get", "route_role"], "alternative_route"]],
      paint: {
        "line-color": "rgba(255,255,255,.92)",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 4.5, 10, 7, 15, 10],
        "line-offset": 3.5,
        "line-opacity": .92,
      },
    });
    this.map.addLayer({
      id: "route-alternative-links",
      type: "line",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["==", ["get", "route_role"], "alternative_route"]],
      paint: {
        "line-color": ["coalesce", ["get", "route_colour"], "#ffd43b"],
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 2.4, 10, 4.2, 15, 6.2],
        "line-offset": 3.5,
        "line-opacity": .94,
      },
    });
    this.map.addLayer({
      id: "route-endpoints",
      type: "circle",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "Point"], ["in", ["get", "route_role"], ["literal", ["route_origin", "route_destination"]]]],
      paint: {
        "circle-radius": 9,
        "circle-color": ["case", ["==", ["get", "route_role"], "route_origin"], "#267a74", "#ffd43b"],
        "circle-stroke-width": 3,
        "circle-stroke-color": "#ffffff",
      },
    });
    this.map.addLayer({
      id: "analysis-link-casing",
      type: "line",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["!=", ["get", "analysis_type"], "route_detour_analysis"]],
      paint: {
        "line-color": "rgba(74,57,0,.82)",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 5.5, 10, 8, 15, 11],
        "line-opacity": .95,
      },
    });
    this.map.addLayer({
      id: "analysis-result-links",
      type: "line",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["!=", ["get", "analysis_type"], "route_detour_analysis"]],
      paint: {
        "line-color": "#ffd43b",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 3, 10, 5, 15, 7],
        "line-opacity": .96,
      },
    });
    this.map.addLayer({
      id: "analysis-area-fill",
      type: "fill",
      source: "analysis-result",
      filter: ["==", ["geometry-type"], "Polygon"],
      paint: {
        "fill-color": ["case", ["==", ["get", "analysis_type"], "spatial_hotspots"], ["interpolate", ["linear"], ["coalesce", ["get", "rank_score"], 0], 0, "#fff3a3", .5, "#f4c542", 1, "#b77900"], ["==", ["get", "analysis_type"], "restoration_screening"], "#ffd43b", ["==", ["get", "analysis_type"], "area_summary"], "#267a74", "#a7354d"],
        "fill-opacity": ["case", ["==", ["get", "analysis_type"], "spatial_hotspots"], .42, .2],
      },
    });
    this.map.addLayer({
      id: "analysis-area-line",
      type: "line",
      source: "analysis-result",
      filter: ["==", ["geometry-type"], "Polygon"],
      paint: { "line-color": ["case", ["==", ["get", "analysis_type"], "spatial_hotspots"], "#b77900", ["==", ["get", "analysis_type"], "restoration_screening"], "#9b6b00", ["==", ["get", "analysis_type"], "area_summary"], "#145d58", "#7f2037"], "line-width": 2.2 },
    });
    this.map.addLayer({
      id: "analysis-halo",
      type: "circle",
      source: "analysis-result",
      filter: ["all", ["==", ["geometry-type"], "Point"], ["!=", ["get", "analysis_type"], "route_detour_analysis"]],
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["coalesce", ["get", "severity"], 0], 0, 9, 1, 20],
        "circle-color": "rgba(167,53,77,.08)",
        "circle-stroke-width": 2.4,
        "circle-stroke-color": "#a7354d",
      },
    });
    this.map.addSource("selected-area", { type: "geojson", data: EMPTY_COLLECTION });
    this.map.addLayer({ id: "selected-area-fill", type: "fill", source: "selected-area", paint: { "fill-color": "#a7354d", "fill-opacity": .12 } });
    this.map.addLayer({ id: "selected-area-line", type: "line", source: "selected-area", paint: { "line-color": "#7f2037", "line-width": 2.4, "line-dasharray": [2, 1.5] } });
    this.map.addSource("scenario-result", { type: "geojson", data: EMPTY_COLLECTION });
    this.map.addLayer({
      id: "scenario-direct-lines",
      type: "line",
      source: "scenario-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["!=", ["get", "impact_role"], "rerouting_pressure"]],
      paint: {
        "line-color": "#f4c542",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 4, 10, 6, 15, 8],
        "line-opacity": .98,
      },
    });
    this.map.addLayer({
      id: "scenario-indirect-lines",
      type: "line",
      source: "scenario-result",
      filter: ["all", ["==", ["geometry-type"], "LineString"], ["==", ["get", "impact_role"], "rerouting_pressure"]],
      paint: {
        "line-color": "#ee8f25",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 3, 10, 5, 15, 7],
        "line-opacity": .98,
        "line-dasharray": [2.2, 1.3],
      },
    });
    this.map.addLayer({
      id: "scenario-result-points",
      type: "circle",
      source: "scenario-result",
      filter: ["==", ["geometry-type"], "Point"],
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"],
          4, ["case", ["==", ["get", "impact_role"], "rerouting_pressure"], 7, 6],
          10, ["case", ["==", ["get", "impact_role"], "rerouting_pressure"], 8.5, 10],
          15, ["case", ["==", ["get", "impact_role"], "rerouting_pressure"], 10, 14]],
        "circle-color": ["case", ["==", ["get", "impact_role"], "rerouting_pressure"], "#ee8f25", "#f4c542"],
        "circle-stroke-width": 2.4,
        "circle-stroke-color": "#fff",
        "circle-opacity": .94,
      },
    });
    this.map.addSource("scenario-draft", { type: "geojson", data: EMPTY_COLLECTION });
    this.map.addLayer({
      id: "scenario-draft-lines",
      type: "line",
      source: "scenario-draft",
      filter: ["==", ["geometry-type"], "LineString"],
      paint: { "line-color": "#f0b34f", "line-width": 10, "line-opacity": .78 },
    });
    this.map.addLayer({
      id: "scenario-draft-points",
      type: "circle",
      source: "scenario-draft",
      filter: ["==", ["geometry-type"], "Point"],
      paint: { "circle-radius": 12, "circle-color": "rgba(240,179,79,.24)", "circle-stroke-width": 3, "circle-stroke-color": "#d98d1f" },
    });
    this.map.addSource("matsim-simulation", { type: "geojson", data: EMPTY_COLLECTION });
    this.map.addLayer({
      id: "matsim-link-casing",
      type: "line",
      source: "matsim-simulation",
      filter: ["==", ["geometry-type"], "LineString"],
      paint: {
        "line-color": "rgba(255,255,255,.94)",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 5, 10, 8, 15, 11],
        "line-opacity": .9,
      },
    });
    this.map.addLayer({
      id: "matsim-link-load",
      type: "line",
      source: "matsim-simulation",
      filter: ["==", ["geometry-type"], "LineString"],
      paint: {
        "line-color": ["interpolate", ["linear"], ["coalesce", ["get", "load_ratio"], 0], 0, "#1f8078", .35, "#f2bf32", .7, "#ef7e29", 1, "#b12f4e"],
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 2.5, 10, 5, 15, 7],
        "line-opacity": .96,
      },
    });
    this.map.addLayer({
      id: "matsim-vehicle-flow-glow",
      type: "circle",
      source: "matsim-simulation",
      filter: ["==", ["geometry-type"], "Point"],
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 7, 10, 12, 15, 16],
        "circle-color": ["match", ["get", "impact_relation"], "blocked", "#7f1734", "changed", "#b12f4e", "nearby", "#f4b400", "#1f8078"],
        "circle-blur": .8,
        "circle-opacity": ["*", .48, ["coalesce", ["get", "playback_opacity"], 1]],
      },
    });
    this.map.addLayer({
      id: "matsim-vehicles",
      type: "circle",
      source: "matsim-simulation",
      filter: ["==", ["geometry-type"], "Point"],
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 3.5, 10, 6, 15, 8],
        "circle-color": ["match", ["get", "impact_relation"], "blocked", "#7f1734", "changed", "#b12f4e", "nearby", "#f4b400", "#1f8078"],
        "circle-stroke-color": "#7d3c00",
        "circle-stroke-width": 1.5,
        "circle-opacity": ["*", .96, ["coalesce", ["get", "playback_opacity"], 1]],
      },
    });
  }

  _installInteractions() {
    this.areaBox = document.createElement("div");
    this.areaBox.className = "map-area-drag-box is-hidden";
    this.areaBox.setAttribute("aria-hidden", "true");
    this.map.getContainer().append(this.areaBox);

    this.map.on("click", "observation-clusters", (event) => {
      if (this.mode === "area") return;
      const feature = event.features?.[0];
      if (!feature) return;
      const clusterId = feature.properties.cluster_id;
      this.map.getSource("observations").getClusterExpansionZoom(clusterId, (error, zoom) => {
        if (error) return;
        this.map.easeTo({ center: feature.geometry.coordinates, zoom });
      });
    });
    this.map.on("mouseenter", "observation-points", () => {
      this.map.getCanvas().style.cursor = this.mode === "area" ? "crosshair" : "pointer";
    });
    this.map.on("mouseleave", "observation-points", () => { this.map.getCanvas().style.cursor = this.mode === "area" ? "crosshair" : ""; });
    this.map.on("mouseenter", "network-links", () => {
      this.map.getCanvas().style.cursor = this.mode === "area" ? "crosshair" : "pointer";
    });
    this.map.on("mouseleave", "network-links", () => { this.map.getCanvas().style.cursor = this.mode === "area" ? "crosshair" : ""; });
    const analysisPopup = new window.maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10 });
    const routeDetailPopup = new window.maplibregl.Popup({ closeButton: true, closeOnClick: true, offset: 12 });
    const showAnalysisPopup = (event, popup = analysisPopup) => {
      const feature = event.features?.[0];
      if (!feature) return;
      const properties = feature.properties || {};
      const content = document.createElement("div");
      content.className = "map-analysis-tooltip";
      const title = document.createElement("strong");
      const routeRole = properties.route_role;
      const routeLine = routeRole && !["route_origin", "route_destination"].includes(routeRole);
      title.textContent = routeRole
        ? routeLine
          ? properties.route_label || "Route"
          : properties.route_label || (routeRole === "route_origin" ? "Start point / road" : "Destination point / road")
        : properties.cell_id ? `Rank #${properties.rank} · ${properties.cell_id}` : `Rank #${properties.rank || "—"} · Link ${properties.entity_id || "result"}`;
      const score = document.createElement("span");
      const dpcValue = properties.regional_dpc_score;
      const value = properties.efficiency_loss ?? properties.structural_score ?? dpcValue ?? properties.mean_degradation;
      if (routeLine) {
        const impedance = Number(properties.route_impedance);
        const unit = String(properties.route_cost_unit || "uploaded unit");
        const timeUnit = /(min|second|hour)/i.test(unit);
        score.textContent = Number.isFinite(impedance)
          ? `${timeUnit ? "Total time" : "Total route cost"}: ${impedance.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${unit}`
          : "Route total unavailable";
      } else {
        score.textContent = routeRole
          ? "Confirmed route endpoint"
          : Number.isFinite(Number(value))
            ? `${Number.isFinite(Number(dpcValue)) ? "Regional DPC loss" : properties.cell_id && properties.mean_degradation != null ? "Observed mean degradation" : "Score"}: ${(Number(value) * 100).toFixed(Number.isFinite(Number(dpcValue)) ? 3 : 2)}%`
            : "Mapped analysis result";
      }
      const method = document.createElement("small");
      if (routeLine) {
        const distance = properties.route_distance_m == null ? Number.NaN : Number(properties.route_distance_m);
        const sharedCount = Number(properties.route_shared_count || 0);
        const segmentMeaning = sharedCount > 1
          ? ` · this road is shared by ${sharedCount.toLocaleString()} recommendations`
          : properties.route_segment_status === "route-specific" ? " · route-specific road" : "";
        method.textContent = `${Number(properties.route_link_count || 0).toLocaleString()} roads${Number.isFinite(distance) ? ` · ${(distance / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 })} km` : ""}${segmentMeaning} · click to keep this information open`;
      } else {
        method.textContent = properties.method_label || properties.ranking_method || "See the complete result for method and limits";
      }
      content.append(title, score, method);
      popup.setLngLat(event.lngLat).setDOMContent(content).addTo(this.map);
    };
    ["analysis-result-links", "analysis-area-fill", "analysis-halo", "route-baseline-links", "route-comparison-links", "route-alternative-links", "route-endpoints"].forEach((layer) => {
      this.map.on("mouseenter", layer, (event) => { this.map.getCanvas().style.cursor = "pointer"; showAnalysisPopup(event); });
      this.map.on("mousemove", layer, showAnalysisPopup);
      this.map.on("mouseleave", layer, () => { analysisPopup.remove(); this.map.getCanvas().style.cursor = this.mode === "area" ? "crosshair" : ""; });
    });
    this.map.on("mousedown", (event) => {
      if (this.mode !== "area") return;
      event.preventDefault?.();
      this.areaDrag = { start: event.point };
      this.map.getSource("selected-area").setData(EMPTY_COLLECTION);
      this._updateAreaDragBox(event.point, event.point);
      if (this.callbacks.onAreaProgress) this.callbacks.onAreaProgress("Drag to size the rectangle, then release.");
    });
    this.map.on("mousemove", (event) => {
      if (this.mode !== "area" || !this.areaDrag) return;
      this._updateAreaDragBox(this.areaDrag.start, event.point);
    });
    const finishAreaDrag = (point) => {
      if (this.mode !== "area" || !this.areaDrag || !point) return;
      const start = this.areaDrag.start;
      this._cancelAreaDrag(false);
      if (Math.abs(point.x - start.x) < 8 || Math.abs(point.y - start.y) < 8) {
        if (this.callbacks.onAreaProgress) this.callbacks.onAreaProgress("Press, drag and release to draw a rectangular area.");
        return;
      }
      const first = this.map.unproject([Math.min(start.x, point.x), Math.min(start.y, point.y)]);
      const second = this.map.unproject([Math.max(start.x, point.x), Math.max(start.y, point.y)]);
      const bbox = [
        Math.min(first.lng, second.lng),
        Math.min(first.lat, second.lat),
        Math.max(first.lng, second.lng),
        Math.max(first.lat, second.lat),
      ];
      this.setArea(bbox);
      if (this.callbacks.onArea) this.callbacks.onArea(bbox);
    };
    this.map.on("mouseup", (event) => {
      finishAreaDrag(event.point);
    });
    // A fixed overlay (for example the AI Assistant launcher) can sit over the
    // map. Complete an in-progress rectangle even when the pointer is released
    // on that overlay or just outside the canvas.
    window.addEventListener("mouseup", (event) => {
      if (this.mode !== "area" || !this.areaDrag) return;
      const rect = this.map.getContainer().getBoundingClientRect();
      finishAreaDrag({
        x: Math.max(0, Math.min(rect.width, event.clientX - rect.left)),
        y: Math.max(0, Math.min(rect.height, event.clientY - rect.top)),
      });
    }, true);
    this.map.on("click", (event) => {
      if (this.mode === "area") return;
      const cluster = this.map.queryRenderedFeatures(event.point, { layers: ["observation-clusters"] });
      if (cluster.length) return;
      const routeFeatures = this.map.queryRenderedFeatures(
        [[event.point.x - 8, event.point.y - 8], [event.point.x + 8, event.point.y + 8]],
        { layers: ["route-alternative-links", "route-comparison-links", "route-baseline-links"] },
      );
      if (routeFeatures.length) {
        analysisPopup.remove();
        showAnalysisPopup({ features: [routeFeatures[0]], lngLat: event.lngLat }, routeDetailPopup);
        return;
      }
      const features = this.map.queryRenderedFeatures(
        [[event.point.x - 7, event.point.y - 7], [event.point.x + 7, event.point.y + 7]],
        { layers: [this.displayMode === "links" ? "network-links" : "observation-points"] }
      );
      if (features.length && this.callbacks.onFeature) this.callbacks.onFeature(features[0]);
    });
    this.map.on("moveend", () => {
      if (this.callbacks.onViewport && this.ready) {
        const bounds = this.map.getBounds();
        this.callbacks.onViewport([bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()]);
      }
    });
  }

  setMode(mode) {
    this.mode = mode;
    this.map.getCanvas().style.cursor = mode === "area" ? "crosshair" : "";
    if (mode === "area") {
      this.map.dragPan.disable();
    } else {
      this._cancelAreaDrag(false);
      this.map.dragPan.enable();
    }
  }

  _updateAreaDragBox(start, current) {
    if (!this.areaBox) return;
    const left = Math.min(start.x, current.x);
    const top = Math.min(start.y, current.y);
    this.areaBox.style.left = `${left}px`;
    this.areaBox.style.top = `${top}px`;
    this.areaBox.style.width = `${Math.abs(current.x - start.x)}px`;
    this.areaBox.style.height = `${Math.abs(current.y - start.y)}px`;
    this.areaBox.classList.remove("is-hidden");
  }

  _cancelAreaDrag(removeArea) {
    this.areaDrag = null;
    if (this.areaBox) {
      this.areaBox.classList.add("is-hidden");
      this.areaBox.style.width = "0";
      this.areaBox.style.height = "0";
    }
    if (removeArea && this.ready) this.map.getSource("selected-area").setData(EMPTY_COLLECTION);
  }

  setArea(bbox) {
    if (!bbox || !this.ready) {
      if (this.ready) this.map.getSource("selected-area").setData(EMPTY_COLLECTION);
      return;
    }
    const [west, south, east, north] = bbox;
    this.map.getSource("selected-area").setData({
      type: "FeatureCollection",
      features: [{
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]] },
        properties: {},
      }],
    });
  }

  clearArea() {
    this._cancelAreaDrag(true);
  }

  setFeatures(collection, displayMode = "points") {
    this.currentFeatures = collection || EMPTY_COLLECTION;
    this.displayMode = displayMode === "links" ? "links" : "points";
    if (this.ready) {
      this.map.getSource("observations").setData(this.displayMode === "points" ? this.currentFeatures : EMPTY_COLLECTION);
      this.map.getSource("network-links").setData(this.displayMode === "links" ? this.currentFeatures : EMPTY_COLLECTION);
    }
    this.setSelectedKeys([...this.selectedKeys]);
  }

  setSelectedKeys(keys) {
    this.selectedKeys = new Set(keys || []);
    if (!this.ready) return;
    const selected = (this.currentFeatures.features || []).filter((feature) => this.selectedKeys.has(String(feature.properties.key)));
    this.map.getSource("selected").setData({ type: "FeatureCollection", features: selected });
  }

  setAnalysisFeatures(collection) {
    this.analysisFeatures = collection || EMPTY_COLLECTION;
    if (!this.ready) return;
    this.map.getSource("analysis-result").setData(this.analysisFeatures);
    this._updateBaseEmphasis();
  }

  setScenarioFeatures(collection) {
    this.scenarioFeatures = collection || EMPTY_COLLECTION;
    if (!this.ready) return;
    this.map.getSource("scenario-result").setData(this.scenarioFeatures);
    this._updateBaseEmphasis();
  }

  setScenarioDraftFeatures(collection) {
    this.scenarioDraftFeatures = collection || EMPTY_COLLECTION;
    if (!this.ready) return;
    this.map.getSource("scenario-draft").setData(this.scenarioDraftFeatures);
  }

  setMatsimFeatures(collection) {
    this.matsimFeatures = collection || EMPTY_COLLECTION;
    if (!this.ready) return;
    this.map.getSource("matsim-simulation").setData(this.matsimFeatures);
    this._updateBaseEmphasis();
  }

  _updateBaseEmphasis() {
    if (!this.ready) return;
    const overlayVisible = Boolean(this.analysisFeatures?.features?.length || this.scenarioFeatures?.features?.length || this.matsimFeatures?.features?.length);
    this.map.setPaintProperty("network-links", "line-opacity", overlayVisible ? .24 : .88);
    this.map.setPaintProperty("network-link-casing", "line-opacity", overlayVisible ? .18 : .9);
    this.map.setPaintProperty("observation-points", "circle-opacity", overlayVisible ? .24 : .88);
    this.map.setPaintProperty("observation-clusters", "circle-opacity", overlayVisible ? .3 : 1);
  }

  setExtent(extent, fit = true) {
    this.lastExtent = extent;
    if (fit) this.fitExtent();
  }

  fitExtent() {
    if (!this.map || !this.lastExtent || this.lastExtent.some((value) => !Number.isFinite(Number(value)))) return;
    const [west, south, east, north] = this.lastExtent.map(Number);
    this.map.fitBounds([[west, south], [east, north]], { padding: 48, duration: 650, maxZoom: 12 });
  }

  focusCollection(collection) {
    const features = collection?.features || [];
    if (!features.length) return;
    const coordinates = [];
    const collectCoordinates = (value) => {
      if (!Array.isArray(value)) return;
      if (value.length >= 2 && Number.isFinite(Number(value[0])) && Number.isFinite(Number(value[1]))) {
        coordinates.push([Number(value[0]), Number(value[1])]);
        return;
      }
      value.forEach(collectCoordinates);
    };
    features.forEach((feature) => collectCoordinates(feature.geometry?.coordinates));
    if (!coordinates.length) return;
    const bounds = coordinates.reduce(
      (box, coordinate) => box.extend(coordinate),
      new window.maplibregl.LngLatBounds(coordinates[0], coordinates[0])
    );
    this.map.fitBounds(bounds, { padding: 70, duration: 650, maxZoom: 13 });
  }

  resize() {
    if (this.map) this.map.resize();
  }
}
