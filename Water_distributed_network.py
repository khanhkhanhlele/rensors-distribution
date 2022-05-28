import argparse
import os
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

class Population:       # quần thể
    def __init__(self, path, size_of_population, num_available_sensors) -> None:
        _, self.water_network, self.min_det_time, self.dt, self.node_to_id, self.n_scenarios, self.sensor_characteristics, self.scenario_characteristics = input_network(path)
        self.scenario_names = self.water_network.node_name_list
        self.sensor_names = self.water_network.junction_name_list
        self.num_sensors = len(self.sensor_names)
        self.size_of_population = size_of_population
        self.num_available_sensors = num_available_sensors
        self.method = self.detection_time_GA
        self.list_individuals = self.init_population()
        self.best_individual = None
        self.best_sensors = None
        self.score = float("inf")
        self.parents = []

    def detection_time_GA(self, impact, selected_sensors):
        sensors_id = [self.node_to_id.get(s) for s in selected_sensors]
        return self.dt[sensors_id, :].min(axis=0).mean()


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
        # log_node_iter.append(self.best_sensors)
        # log_population.append(self.list_individuals)
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

        #self.parents = np.asarray(self.parents)

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

        if not (_next == self.best_individual).any():
            _next[-1] = self.best_individual
        return _next


    def init_population(self):
    # khởi tạo 1 cá thể ngẫu nhiên
        def init_individual(sensor_names, num_available_sensors):
            
            individual = np.random.choice(range(len(sensor_names)), size=num_available_sensors, replace=False)
            individual.sort()
            return individual

        sensor_names = self.water_network.junction_name_list
        population = []
        for id in range(0, self.size_of_population):
            population.append( init_individual(sensor_names, self.num_available_sensors) )
        return population

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

def genetic_algorithm(path, size_of_population, num_available_sensors, selectivity=0.15, n_iter=100,
    p_cross=0.75, p_mut=0.1, print_interval=100, return_history=False, verbose=False):
    pop = Population(path, size_of_population, num_available_sensors)
    best_sensors = pop.best_sensors
    score = float("inf")
    history = []
    for i in range(n_iter):
        pop.select(size_of_population * selectivity)
        # print(pop.parents)
        history.append(pop.score)
        
        if pop.score < score:
            best_sensors = pop.best_sensors
            score = pop.score
        
        children = pop.next_population(p_cross, p_mut)
        
        pop.list_individuals = children
    if return_history:
        return best_sensors, score, history
    return best_sensors, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='path to network file')
    time_s = time.time()
    args = parser.parse_args()
    path = args.path
    if not os.path.isdir(path): print("Wrong path")
    elif path[-3:0] == "inp":
        best, score, history = genetic_algorithm(path = path, size_of_population=n_scenarios*20, num_available_sensors=5, n_iter=100,
            selectivity=0.15, p_cross=0.75, p_mut=0.1, print_interval=5, verbose=True, return_history=True)
        GA_time = time.time() - time_s
        print('GA time',GA_time)
        print('GA result', score)
    else:
        for i, p in enumerate(os.listdir(path)):
            best, score, history = genetic_algorithm(path = p, size_of_population=n_scenarios*20, num_available_sensors=5, n_iter=100,
            selectivity=0.15, p_cross=0.75, p_mut=0.1, print_interval=5, verbose=True, return_history=True)
            GA_time = time.time() - time_s
            print("{}.Network file: {}".format(i, p))
            print('GA time',GA_time)
            print('GA result', score)
            print("----------------------------------")