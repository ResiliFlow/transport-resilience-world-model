# Third-party notices

Copyright © The University of New South Wales (UNSW Sydney) 2026. Released under the MIT License.

The root licence covers original project code, not the third-party materials listed below. Product names and institutional logos identify their owners; the MIT licence does not grant trademark rights or imply endorsement.

## Bundled software

**MapLibre GL JS 5.6.0** is distributed under the BSD 3-Clause licence, with additional notices for its incorporated code. The full upstream notice is included in [frontend/vendor/LICENSE.maplibre.txt](frontend/vendor/LICENSE.maplibre.txt). The same notice applies to the copy used by the road-safety interface.

## Optional software and services

- **MATSim:** GPL v2. No MATSim binaries are bundled. Building or distributing the optional Java worker with MATSim requires compliance with MATSim and its dependencies' licences. See [MATSim](https://github.com/matsim-org/matsim-libs).
- **Meerut reference:** the worker architecture was informed by `ashutoshkaushal1/autocalibration_meerut` at commit `81dcc140`, identified in the original project as GPL v2. No reference-project data or location-specific implementation is included.
- **Ultralytics:** optional detector workers use Ultralytics APIs. Ultralytics has [AGPL-3.0 and Enterprise licensing options](https://www.ultralytics.com/license). No Ultralytics package or trained weights are bundled; the root MIT licence does not override its terms.
- **Other Python and Java dependencies:** installed separately and retain their own licences.

## Map and image sources

- **OpenStreetMap:** map data © OpenStreetMap contributors, under the [Open Database License](https://www.openstreetmap.org/copyright).
- **Google Maps and Street View:** subject to [Google Maps Platform terms](https://cloud.google.com/maps-platform/terms). No Google imagery or credentials are distributed.
- **KartaView:** imagery attribution and permitted uses are governed by [KartaView's terms](https://kartaview.org/terms), including the applicable CC BY-SA conditions.
- **Copernicus Sentinel:** subject to the [Copernicus Data Space terms](https://dataspace.copernicus.eu/terms-and-conditions).
- **Wikimedia Commons:** each image has its own licence and attribution requirements.

Only synthetic CSV fixtures are included as example data. The repository does not grant permission to redistribute or train on external data or imagery.
