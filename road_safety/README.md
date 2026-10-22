# Road-safety workspace

This is a separate local interface for reviewing road-safety results and preparing inspection lists. It does not run the main platform's detector or train a model.

The research data bundle is not public and is not included. The source expects a `bundle/` directory containing its manifest, result tables and map geometry. Do not replace these with invented results.

With an authorised compatible bundle installed, run:

```sh
python server.py --port 8010
```

Open [localhost:8010/safety](http://127.0.0.1:8010/safety). Without the bundle, the module reports that its data is unavailable.

Contributions to the interface and calculation code are welcome. Use synthetic fixtures when proposing changes.
