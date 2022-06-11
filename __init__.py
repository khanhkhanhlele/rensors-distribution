import os
import run_by_chama, run_by_ga, input_network, run_pso, read_network
import numpy as np
import pandas as pd
import shutil
import csv

path = "data_read_success"
dir_list = os.listdir(path)
for a in dir_list:
  if 'inp' in a:
    try:
        path = "data_read_success/" + a
        des = "data_high_nodes/"+a
        info = read_network.read_network(path)
        if info[1]>=500:
            shutil.copyfile(path, des)
    except:
        print('fail:-------------------------', a)


# path = "data_read_success"
# dir_list = os.listdir(path)
# f = open('result/table.csv','w')
# writer = csv.writer(f)
# for a in dir_list:
#     try:
#         path = "data_read_success/"+a
#         info = read_network.read_network(path)
#         writer.writerow(info)
#     except:
#         print('fail:-------------------------', a)
# f.close()



# read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics = input_network.input_network(path)
# limit_sensors = 6
# chama_result, chama_time = run_by_chama.run_by_chama(
#     min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors)

# PSO_time, PSO_score, PSO_pos = run_pso.run_pso(
#     read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics, limit_sensors)


# print([[net, limit_sensors, net_info['Nodes'], net_info['Links'], 
#     read_time, chama_time, chama_result['Objective'], PSO_time, PSO_score]])
