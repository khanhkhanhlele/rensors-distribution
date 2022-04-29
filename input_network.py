import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import chama
import wntr
import time


def input_network(path):
    time_s = time.time()
    water_network = wntr.network.WaterNetworkModel(path)
    # Run trace simulations (one from each junction) and extract data needed for
    # sensor placement optimization. You can run this step once, save the data to a
    # file, and reload the file for sensor placement
    scenario_names = water_network.node_name_list
    sim = wntr.sim.EpanetSimulator(water_network)
    sim.run_sim(save_hyd=True)
    water_network.options.quality.parameter = 'TRACE'
    signal = pd.DataFrame()
    for inj_node in scenario_names:
        water_network.options.quality.trace_node = inj_node
        sim_results = sim.run_sim(use_hyd=True)
        trace = sim_results.node['quality']
        trace = trace.stack()
        trace = trace.reset_index()
        trace.columns = ['T', 'Node', inj_node]
        signal = signal.combine_first(trace)
    # Define feasible sensors using location, sample times, and detection threshold

    sensor_names = water_network.node_name_list
    sample_times = np.arange(0, water_network.options.time.duration,
                             water_network.options.time.hydraulic_timestep)
    undetected_impact = sample_times.max()*1.5
    node_to_id = dict(zip(scenario_names, np.arange(len(scenario_names))))
    id_to_node = dict(zip(np.arange(len(scenario_names)), scenario_names))
    n_scenarios = len(scenario_names)

    df_dummy = pd.DataFrame({'Scenario': scenario_names,
                            'Sensor': 'DUMMY_SENSOR_UNDETECTED',
                             'Impact': undetected_impact})
    threshold = 1e-5
    # threshold = 20
    sensors = {}
    for location in sensor_names:
        position = chama.sensors.Stationary(location)
        detector = chama.sensors.Point(threshold, sample_times)
        stationary_pt_sensor = chama.sensors.Sensor(position, detector)
        sensors[location] = stationary_pt_sensor

    # Extract minimum detection time for each scenario-sensor pair
    det_times = chama.impact.extract_detection_times(signal, sensors)
    det_time_stats = chama.impact.detection_time_stats(det_times)
    min_det_time = det_time_stats[['Scenario', 'Sensor', 'Min']]
    min_det_time.rename(columns={'Min': 'Impact'}, inplace=True)

    scenario_characteristics = pd.DataFrame({'Scenario': scenario_names,
                                             'Undetected Impact': undetected_impact})
    sensor_characteristics = pd.DataFrame({'Sensor': sensor_names, 'Cost': 1})

    cols = min_det_time.loc[:, "Scenario"].apply(node_to_id.get).values
    rows = min_det_time.loc[:, "Sensor"].apply(node_to_id.get).values
    values = min_det_time.loc[:, "Impact"]

    dt = np.full((rows.max() + 1, cols.max() + 1), undetected_impact)
    dt[rows, cols] = values
    read_time = time.time()-time_s
    return (read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics)
