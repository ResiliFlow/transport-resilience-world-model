# Contributing

This is an experimental research project. Useful contributions include bug fixes, clearer interfaces, data import improvements, tests and new transport-analysis methods.

For a substantial change, open an issue first so we can agree on the scope.

## Development

Follow the local setup in the README, then install the test dependency:

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The automated tests use synthetic data and mocked provider calls. They do not establish real-world detector accuracy or verify live paid APIs.

The MATSim integration test runs only when Java and the optional worker JAR are installed; otherwise it is reported as skipped.

The main folders are:

- `backend/`: application endpoints and project storage.
- `engine/`: transport analysis and assistant logic.
- `frontend/`: browser interface.
- `model_library/`: analysis methods and their references.
- `infrastructure_perception/`: image-review and detector integration source.
- `matsim_worker/`: optional Java simulation worker.
- `road_safety/`: independent road-safety interface.

## Pull requests

Explain the problem, keep the change focused and include a test where practical. Tell us which checks you ran and any limitations.

Use synthetic or openly licensed examples. Do not include API keys, personal data, saved projects, chat transcripts, work logs, private experiments or model weights.

Keep existing licences and attribution. By submitting code, you confirm you have the right to contribute it under the project's licence.
