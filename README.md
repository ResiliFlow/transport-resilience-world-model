# Transport Resilience World Model

An experimental platform for exploring how transport networks respond to disruption and how they can recover.

Developed within **ResiliFlow**, this project brings maps, road-condition review, transport analysis and recovery planning into one workspace. It is shared for research and collaborative development, not as a finished service or a certified decision-making tool.

[Watch the platform video on YouTube](https://www.youtube.com/watch?v=3QHWsOzBRiw) · [Read the arXiv preprint](https://arxiv.org/abs/2608.20709)

## What you can work on

- Import transport data and explore it on a map.
- Compare normal conditions with road closures and other disruptions.
- Examine important roads, alternative routes and recovery priorities.
- Review infrastructure images and human corrections.
- Extend the analysis tools, simulation connections and optional AI assistant.

The code supports these workflows, but their results depend on the available data and models. Image detection remains experimental; this release makes no citywide accuracy claim.

## What is included

The application source, analysis models, interface, synthetic test data and automated tests are included.

Model weights, research datasets, saved projects and service credentials are **not** included. Image detection needs a separately licensed, compatible model package. The road-safety workspace needs its own authorised data bundle.

This repository does not provide access to a hosted backend or to the maintainers' private platform.

## Run locally

Use Python 3.11 or 3.12. From the repository folder:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux, then run:

```sh
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8005
```

Open [localhost:8005](http://127.0.0.1:8005). Upload one of the small CSV examples in `data/synthetic_acceptance/` to explore the import workflow. These examples are artificial test data, not observations or evidence of model accuracy.

The application is intended for local development. Do not expose it directly to the public internet without adding authentication and reviewing data access.

## Optional features

- **Maps and imagery:** external providers have their own terms and attribution requirements. Only analyse imagery where your rights and the provider's terms allow it.
- **AI assistant:** supply your own provider key through the application. Do not commit keys or paste them into GitHub issues.
- **Image detection:** source code is included, but trained weights and their licences must be supplied separately. Missing models must not be reported as working detectors.
- **MATSim:** the optional Java worker requires Java 21, Maven and MATSim. See [the worker guide](matsim_worker/README.md).
- **Road safety:** see [the module guide](road_safety/README.md) for its separate data requirements.

## Contribute

Open an issue to discuss an idea, report a reproducible problem or propose a collaboration. Small, focused pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and checks.

## Licence

Copyright © The University of New South Wales (UNSW Sydney) 2026. Released under the MIT License.

The [MIT licence](LICENSE) applies to the original project code. Third-party software, imagery, data, model weights and branding have separate terms; see [NOTICE.md](NOTICE.md).
