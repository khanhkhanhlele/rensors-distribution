# -*- coding: utf-8 -*-
"""sensor opt compare.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/khanhkhanhlele/rensors-distribution/blob/main/sensor_opt_compare.ipynb

## Import
"""

# !apt-get install -y -qq glpk-utils
# !pip install -qqq chama wntr pyswarms
# !pip install pyswarms

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import chama
import wntr
import random
import time
import math
import pyswarms as ps
from pyswarms.discrete import BinaryPSO
from scipy.special import softmax
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from numpy.random import choice

"""#INITIALIZE

##Visual Network
"""

def read_network(path):
    water_network = wntr.network.WaterNetworkModel(path)
    return water_network.describe(level=0)

"""##Read File"""

def input_network(path):
    time_s = time.time()
    water_network = wntr.network.WaterNetworkModel(path)
    # Run trace simulations (one from each junction) and extract data needed for 
    # sensor placement optimization. You can run this step once, save the data to a 
    # file, and reload the file for sensor placement
    scenario_names = water_network.node_name_list
    sim = wntr.sim.EpanetSimulator(water_network)
    sim.run_sim(save_hyd = True)
    water_network.options.quality.parameter = 'TRACE'
    signal = pd.DataFrame()
    for inj_node in scenario_names:
        water_network.options.quality.trace_node = inj_node
        sim_results = sim.run_sim(use_hyd = True)
        trace = sim_results.node['quality']
        trace = trace.stack()
        trace = trace.reset_index()
        trace.columns = ['T', 'Node', inj_node]
        signal = signal.combine_first(trace)
    # Define feasible sensors using location, sample times, and detection threshold

    sensor_names = water_network.node_name_list
    sample_times = np.arange(0, water_network.options.time.duration, water_network.options.time.hydraulic_timestep)
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
    min_det_time = det_time_stats[['Scenario','Sensor','Min']]
    min_det_time.rename(columns = {'Min':'Impact'}, inplace = True)

    scenario_characteristics = pd.DataFrame({'Scenario': scenario_names,
                                        'Undetected Impact': undetected_impact})
    sensor_characteristics = pd.DataFrame({'Sensor': sensor_names,'Cost': 1})

    cols = min_det_time.loc[:, "Scenario"].apply(node_to_id.get).values
    rows = min_det_time.loc[:, "Sensor"].apply(node_to_id.get).values
    values = min_det_time.loc[:, "Impact"]

    dt = np.full((rows.max() + 1, cols.max() + 1), undetected_impact)
    dt[rows, cols] = values
    read_time = time.time()-time_s
    return (read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics)

"""##Chama"""

def run_by_chama(min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors):
  time_s = time.time()
  impactform = chama.optimize.ImpactFormulation()
  result = impactform.solve(min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors)
  #wntr.graphics.plot_network(water_network, node_attribute=result['Sensors'])
  chama_time = time.time()-time_s
  return result, chama_time

"""##GA"""

log_node_iter = []
log_population = []

class Population:       # quần thể
    def __init__(self, water_network, size_of_population, num_available_sensors, method, list_individuals) -> None:
        self.water_network = water_network
        self.scenario_names = water_network.node_name_list
        self.sensor_names = water_network.junction_name_list
        self.num_sensors = len(self.sensor_names)
        self.size_of_population = size_of_population
        self.num_available_sensors = num_available_sensors
        self.method = method
        self.list_individuals = list_individuals
        self.best_individual = None
        self.best_sensors = None
        self.score = 0
        self.parents = []


    def choosed_sensors(self, individual):      # from idx -> list sensor
        try:
            _choosed = [self.sensor_names[i] for i in individual]
        except:
            print(self.sensor_names[i])
            print(individual)
            input()
        return _choosed

    # đột biến
    def mutate(self, individual):
        def _mutate():
            new_sensor = random.choice(range(self.num_sensors))
            while new_sensor in individual:
                new_sensor = random.choice(range(self.num_sensors))
            idx = np.random.randint(len(individual))
            individual[idx] = new_sensor
            individual.sort()
            return individual

        _new = _mutate()
        return _new

    def evaluate(self):
        times = np.asarray([ self.method(min_det_time, self.choosed_sensors(individual)) for individual in self.list_individuals])
        self.score = np.min(times)
       
        self.best_individual = self.list_individuals[times.tolist().index(self.score)]
        
        self.best_sensors = self.choosed_sensors(self.best_individual)
        log_node_iter.append(self.best_sensors)
        log_population.append(self.list_individuals)
        self.parents.append(self.best_individual)
        if False in (times[0] == times):
            distances = np.max(times) - times
        return times / np.sum(times)


    def select(self, num_parents):
        time_s = time.time()
        fit = self.evaluate()
#         print("Time eval: ", time.time() - time_s)
        while len(self.parents) < num_parents:
            idx = np.random.randint(0, self.size_of_population)
            if fit[idx] > np.random.uniform(0, 1/self.size_of_population, size=(1,))[0]:
                self.parents.append(self.list_individuals[idx])

        self.parents = np.asarray(self.parents)

    def crossover(self, p_cross=0.75):
        def cross():
            id1, id2 = np.random.choice(len(self.parents), size=2, replace=False)
            parent1, parent2 = self.parents[id1], self.parents[id2]
           # print(parent1)
           # print(parent2)
            gen_set=set(np.concatenate((parent1, parent2)))
            gen_set=list(gen_set)
            child = choice(gen_set, size=(self.num_available_sensors,), replace=False)
            child.sort()
            #print(child)
            return child
            
        children = []
        count = len(self.parents)
        for _ in range(self.size_of_population):
            if np.random.rand() > p_cross:
                _tmp = random.choice(self.parents)
                children.append(_tmp)
                
            else:
                child = cross()
                children.append(child)

        # print(children)
        return children

    def next_population(self, p_cross=0.75, p_mutate=0.1):
        _best_individual = self.best_individual
        
        _next = []
        children = self.crossover(p_cross)
        for child in children:
            # print(child.selected_sensors)
            if np.random.rand() < p_mutate:
                # print("mutate")
                child_mutate = self.mutate(child)
                # print(child_mutate)
                if self.method(min_det_time, self.choosed_sensors(_best_individual))  >  self.method(min_det_time, self.choosed_sensors(child_mutate)):
                    _next.append(child_mutate)
                else:
                    _next.append(_best_individual)
                    _best_individual = child_mutate

            else:
                # print("un_mutate")
                # print(child)
                if self.method(min_det_time, self.choosed_sensors(_best_individual))  >  self.method(min_det_time, self.choosed_sensors(child)):
                    _next.append(child)
                else:
                    _next.append(_best_individual)
                    _best_individual = child

        if(self.best_individual not in _next):
            _next[-1] = self.best_individual
        return _next


def init_population(water_network, size_of_population, num_available_sensors, method):
    # khởi tạo 1 cá thể ngẫu nhiên
    def init_individual(sensor_names, num_available_sensors):
        
        individual = np.random.choice(range(len(sensor_names)), size=num_available_sensors, replace=False)
        individual.sort()
        return individual

    sensor_names = water_network.junction_name_list
    population = []
    for id in range(0, size_of_population):
        population.append( init_individual(sensor_names, num_available_sensors) )
    return Population(water_network, size_of_population, num_available_sensors, method, population )

def detection_time_GA(impact, selected_sensors):
    sensors_id = [node_to_id.get(s) for s in selected_sensors]
    return dt[sensors_id, :].min(axis=0).mean()

def genetic_algorithm(
    water_network,
    size_of_population,
    num_available_sensors,
    method,
    selectivity=0.15,
    n_iter=100,
    p_cross=0.75,
    p_mut=0.1,
    print_interval=100,
    return_history=False,
    verbose=False,
):
    t1 = time.time()
    pop = init_population(water_network, size_of_population, num_available_sensors, method)
    best_sensors = pop.best_sensors
    score = float("inf")
    history = []
    for i in range(n_iter):
        pop.select(size_of_population * selectivity)
        # print(pop.parents)
        history.append(pop.score)
        # if verbose:
        #     print(f"Generation {i}: {pop.score}")
        # elif i % print_interval == 0:
        #     print(f"Generation {i}: {pop.score}")
        if pop.score < score:
            best_sensors = pop.best_sensors
            score = pop.score
        
        children = pop.next_population(p_cross, p_mut)
        
        pop = Population(water_network, size_of_population, num_available_sensors, method, children)
    if return_history:
        return best_sensors, score, history
    return best_sensors, score

def run_ga():
  log_node_iter = []
  log_population = []

  time_s = time.time()
  best, score, history = genetic_algorithm(
      water_network = water_network,
      size_of_population=n_scenarios*20,
      num_available_sensors=limit_sensors,
      method=detection_time_GA,
      n_iter=1000,
      selectivity=0.15,
      p_cross=0.75,
      p_mut=0.1,
      print_interval=5,
      verbose=True,
      return_history=True
  )
  GA_time = time.time() - time_s
  #plt.plot(range(len(history)), history, color="blue")
  #wntr.graphics.plot_network(water_network, node_attribute=best)
  #plt.show()

  return GA_time, score, best

"""##PSO"""

def run_pso():
  options = {'c1': 0.5, 'c2': 0.5, 'w':0.9}
  bounds = (-2000. * np.ones(n_scenarios), 2000. * np.ones(n_scenarios))
  time_s = time.time()
  # Call instance of PSO
  optimizer = ps.single.GlobalBestPSO(n_particles = int(n_scenarios*2), dimensions=n_scenarios, options=options, bounds=bounds)

  # Objective function
  def detection_time_PSO(swarms):
      sensor_ids = np.argpartition(swarms, -limit_sensors, axis=1)[:, -5:]  # (n_particles, n_sensors)
      output = dt[sensor_ids].min(axis=1).mean(axis=1)
      return output

  best_cost, best_pos = optimizer.optimize(detection_time_PSO, iters=1000)
  PSO_time = time.time() - time_s
  #plot_cost_history(optimizer.cost_history)
  #plt.show()
  # position
  best_pos = np.argpartition(best_pos, -limit_sensors)[-limit_sensors:]
  PSO_pos = [water_network.node_name_list[a] for a in best_pos]
  # draw
  #wntr.graphics.plot_network(water_network, node_attribute=PSO_pos)
  return  PSO_time, best_cost, PSO_pos

"""#test"""

# cd rensors-distribution



net = "Net1.inp"
path = "data/" + net
net_info = read_network(path)
read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics = input_network(path)
limit_sensors = 6
chama_result, chama_time = run_by_chama(min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors)
GA_time, GA_score, GA_best = run_ga()
PSO_time, PSO_score, PSO_pos = run_pso()
data = np.array([[net, limit_sensors, net_info['Nodes'], net_info['Links'],
                  read_time, chama_time, chama_result['Objective'],
                  GA_time, GA_score, PSO_time, PSO_score]])
#data = np.concatenate((data, new_data), axis=0)
df = pd.DataFrame(data, columns=['Net', 'sensors', 'Nodes', 'Links','read_time', 'chama_time', 'chama_score',
                                 'GA_time', 'GA_score', 'PSO_time', 'PSO_score'])
df.to_csv('result/table.csv', mode='a', header=False)
print(df)