# MATSim worker

This optional Java process connects the platform to MATSim. The main application can start without it.

Use Java 21 and Maven. From this folder, run:

```sh
mvn package
```

The platform looks for `target/twm-matsim-worker.jar`. Simulation also requires a suitable transport network and demand inputs.

No compiled worker or MATSim dependency is distributed in this repository. MATSim is GPL v2; review its licence and all dependency notices before distributing a built worker. The project-wide MIT licence does not replace those requirements. See [the third-party notices](../NOTICE.md).
