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
        times = np.asarray([self.method(min_det_time, self.choosed_sensors(
            individual)) for individual in self.list_individuals])
        self.score = np.min(times)

        self.best_individual = self.list_individuals[times.tolist().index(
            self.score)]

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
            id1, id2 = np.random.choice(
                len(self.parents), size=2, replace=False)
            parent1, parent2 = self.parents[id1], self.parents[id2]
           # print(parent1)
           # print(parent2)
            gen_set = set(np.concatenate((parent1, parent2)))
            gen_set = list(gen_set)
            child = choice(gen_set, size=(
                self.num_available_sensors,), replace=False)
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
                if self.method(min_det_time, self.choosed_sensors(_best_individual)) > self.method(min_det_time, self.choosed_sensors(child_mutate)):
                    _next.append(child_mutate)
                else:
                    _next.append(_best_individual)
                    _best_individual = child_mutate

            else:
                # print("un_mutate")
                # print(child)
                if self.method(min_det_time, self.choosed_sensors(_best_individual)) > self.method(min_det_time, self.choosed_sensors(child)):
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

        individual = np.random.choice(
            range(len(sensor_names)), size=num_available_sensors, replace=False)
        individual.sort()
        return individual

    sensor_names = water_network.junction_name_list
    population = []
    for id in range(0, size_of_population):
        population.append(init_individual(sensor_names, num_available_sensors))
    return Population(water_network, size_of_population, num_available_sensors, method, population)


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
    pop = init_population(water_network, size_of_population,
                          num_available_sensors, method)
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

        pop = Population(water_network, size_of_population,
                         num_available_sensors, method, children)
    if return_history:
        return best_sensors, score, history
    return best_sensors, score


def run_ga():
  log_node_iter = []
  log_population = []

  time_s = time.time()
  best, score, history = genetic_algorithm(
      water_network=water_network,
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
