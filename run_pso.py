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

def run_pso():
  options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
  bounds = (-2000. * np.ones(n_scenarios), 2000. * np.ones(n_scenarios))
  time_s = time.time()
  # Call instance of PSO
  optimizer = ps.single.GlobalBestPSO(n_particles=int(
      n_scenarios*2), dimensions=n_scenarios, options=options, bounds=bounds)

  # Objective function
  def detection_time_PSO(swarms):
      sensor_ids = np.argpartition(
          swarms, -limit_sensors, axis=1)[:, -5:]  # (n_particles, n_sensors)
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
  return PSO_time, best_cost, PSO_pos
