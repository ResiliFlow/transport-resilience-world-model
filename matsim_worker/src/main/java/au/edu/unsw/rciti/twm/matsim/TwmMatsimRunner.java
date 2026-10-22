package au.edu.unsw.rciti.twm.matsim;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.RoutingConfigGroup.NetworkRouteConsistencyCheck;
import org.matsim.core.config.groups.ReplanningConfigGroup.StrategySettings;
import org.matsim.core.config.groups.ScoringConfigGroup.ActivityParams;
import org.matsim.core.controler.Controler;
import org.matsim.core.controler.OutputDirectoryHierarchy.OverwriteFileSetting;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule.DefaultSelector;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule.DefaultStrategy;

/**
 * Minimal, generic MATSim process entrypoint used by the platform backend.
 *
 * <p>The worker deliberately contains no location-specific, hazard or network
 * constants. Python prepares one immutable network and population package per
 * job, then this process runs standard MATSim and writes standard MATSim
 * outputs for independent parsing and verification.</p>
 */
public final class TwmMatsimRunner {
    private TwmMatsimRunner() { }

    public static void main(String[] args) {
        Map<String, String> options = parseArgs(args);
        Path network = requiredPath(options, "network");
        Path plans = requiredPath(options, "plans");
        Path output = requiredPath(options, "output");
        int iterations = positiveInt(options.getOrDefault("iterations", "3"), "iterations");
        long seed = Long.parseLong(options.getOrDefault("seed", "41729"));
        double sampleFactor = positiveDouble(options.getOrDefault("sample-factor", "1.0"), "sample-factor");

        Config config = ConfigUtils.createConfig();
        config.network().setInputFile(network.toAbsolutePath().toString());
        config.plans().setInputFile(plans.toAbsolutePath().toString());
        config.controller().setOutputDirectory(output.toAbsolutePath().toString());
        config.controller().setLastIteration(iterations);
        config.controller().setOverwriteFileSetting(OverwriteFileSetting.deleteDirectoryIfExists);
        config.controller().setCreateGraphs(false);
        config.global().setCoordinateSystem("EPSG:4326");
        config.global().setRandomSeed(seed);
        // Disaster scenarios may deliberately split a directed network. Keep
        // MATSim running so affected agents are recorded as stuck instead of
        // rejecting the scenario before the queue simulation starts.
        config.routing().setNetworkRouteConsistencyCheck(NetworkRouteConsistencyCheck.disable);
        config.global().setNumberOfThreads(Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
        config.qsim().setNumberOfThreads(Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
        config.qsim().setFlowCapFactor(sampleFactor);
        config.qsim().setStorageCapFactor(sampleFactor);
        config.replanning().addStrategySettings(
            new StrategySettings().setStrategyName(DefaultStrategy.ReRoute).setWeight(0.2)
        );
        config.replanning().addStrategySettings(
            new StrategySettings().setStrategyName(DefaultSelector.ChangeExpBeta).setWeight(0.8)
        );
        config.scoring().addActivityParams(new ActivityParams("home").setTypicalDuration(12 * 3600.0));
        config.scoring().addActivityParams(new ActivityParams("work").setTypicalDuration(8 * 3600.0));

        Scenario scenario = ScenarioUtils.loadScenario(config);
        Controler controler = new Controler(scenario);
        controler.run();
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> output = new HashMap<>();
        for (int index = 0; index < args.length; index++) {
            String token = args[index];
            if (!token.startsWith("--") || index + 1 >= args.length) {
                throw new IllegalArgumentException("Expected --name value arguments; received: " + token);
            }
            output.put(token.substring(2), args[++index]);
        }
        return output;
    }

    private static Path requiredPath(Map<String, String> options, String name) {
        String value = options.get(name);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing required --" + name + " path.");
        }
        return Path.of(value);
    }

    private static int positiveInt(String value, String name) {
        int parsed = Integer.parseInt(value);
        if (parsed < 1) throw new IllegalArgumentException(name + " must be positive.");
        return parsed;
    }

    private static double positiveDouble(String value, String name) {
        double parsed = Double.parseDouble(value);
        if (!(parsed > 0.0 && parsed <= 1.0)) {
            throw new IllegalArgumentException(name + " must be in (0, 1].");
        }
        return parsed;
    }
}
