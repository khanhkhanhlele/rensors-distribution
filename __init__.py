import run_by_chama, run_ga, input_network, run_pso, read_network
import numpy as np
import panda as pd

net = "Net1.inp"
path = "data/" + net
net_info = read_network.read_network(path)
read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics = input_network.input_network(path)
limit_sensors = 6
chama_result, chama_time = run_by_chama.run_by_chama(
    min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors)
GA_time, GA_score, GA_best = run_ga.run_ga(
    read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics, limit_sensors)
PSO_time, PSO_score, PSO_pos = run_pso.run_pso(
    read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics, limit_sensors)
data = np.array([[net, limit_sensors, net_info['Nodes'], net_info['Links'],
                  read_time, chama_time, chama_result['Objective'],
                  GA_time, GA_score, PSO_time, PSO_score]])
#data = np.concatenate((data, new_data), axis=0)
df = pd.DataFrame(data, columns=['Net', 'sensors', 'Nodes', 'Links', 'read_time', 'chama_time', 'chama_score',
                                 'GA_time', 'GA_score', 'PSO_time', 'PSO_score'])
df.to_csv('result/table.csv', mode='a', header=False)
print(df)
