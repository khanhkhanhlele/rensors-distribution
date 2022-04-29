import time
import chama

def run_by_chama(min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors):
  time_s = time.time()
  impactform = chama.optimize.ImpactFormulation()
  result = impactform.solve(
      min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors)
  #wntr.graphics.plot_network(water_network, node_attribute=result['Sensors'])
  chama_time = time.time()-time_s
  return result, chama_time
