{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S-5P5z3kfodv"
      },
      "source": [
        "## Import"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-23matlFFWVS"
      },
      "outputs": [],
      "source": [
        "!apt-get install -y -qq glpk-utils\n",
        "!pip install -qqq chama wntr pyswarms\n",
        "!pip install pyswarms"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iJ-DQ_ZUF0N4"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pylab as plt\n",
        "import chama\n",
        "import wntr\n",
        "import random\n",
        "import time\n",
        "import math\n",
        "import pyswarms as ps\n",
        "from pyswarms.discrete import BinaryPSO\n",
        "from scipy.special import softmax\n",
        "from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface\n",
        "from numpy.random import choice"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tCfy64y4Fdqg"
      },
      "source": [
        "#INITIALIZE"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "u69Ki2doF2Nn"
      },
      "source": [
        "##Visual Network"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "YkqbTuHrITsP"
      },
      "outputs": [],
      "source": [
        "def read_network(path):\n",
        "    water_network = wntr.network.WaterNetworkModel(path)\n",
        "    return water_network.describe(level=0)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7WC6NkVZ_B9T",
        "outputId": "c3a02f92-0e90-4908-85f8-48a9ddde5bba"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2022-04-29 04:09:39,115 - wntr.epanet.io - WARNING - Unknown report parameter: MESSAGES\n",
            "2022-04-29 04:09:39,121 - wntr.epanet.io - WARNING - Unknown report parameter: State\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'Controls': 2,\n",
              " 'Curves': 1,\n",
              " 'Links': 13,\n",
              " 'Nodes': 11,\n",
              " 'Patterns': 1,\n",
              " 'Sources': 0}"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ],
      "source": [
        "path = \"rensors-distribution/data/Net1_temp.inp\"\n",
        "read_network(path)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GMqBZKD07WDi"
      },
      "source": [
        "##Read File"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "oxQqwMJU7UXx"
      },
      "outputs": [],
      "source": [
        "def input_network(path):\n",
        "    time_s = time.time()\n",
        "    water_network = wntr.network.WaterNetworkModel(path)\n",
        "    # Run trace simulations (one from each junction) and extract data needed for \n",
        "    # sensor placement optimization. You can run this step once, save the data to a \n",
        "    # file, and reload the file for sensor placement\n",
        "    scenario_names = water_network.node_name_list\n",
        "    sim = wntr.sim.EpanetSimulator(water_network)\n",
        "    sim.run_sim(save_hyd = True)\n",
        "    water_network.options.quality.parameter = 'TRACE'\n",
        "    signal = pd.DataFrame()\n",
        "    for inj_node in scenario_names:\n",
        "        water_network.options.quality.trace_node = inj_node\n",
        "        sim_results = sim.run_sim(use_hyd = True)\n",
        "        trace = sim_results.node['quality']\n",
        "        trace = trace.stack()\n",
        "        trace = trace.reset_index()\n",
        "        trace.columns = ['T', 'Node', inj_node]\n",
        "        signal = signal.combine_first(trace)\n",
        "    # Define feasible sensors using location, sample times, and detection threshold\n",
        "\n",
        "    sensor_names = water_network.node_name_list\n",
        "    sample_times = np.arange(0, water_network.options.time.duration, water_network.options.time.hydraulic_timestep)\n",
        "    undetected_impact = sample_times.max()*1.5\n",
        "    node_to_id = dict(zip(scenario_names, np.arange(len(scenario_names))))\n",
        "    id_to_node = dict(zip(np.arange(len(scenario_names)), scenario_names))\n",
        "    n_scenarios = len(scenario_names)\n",
        "\n",
        "    df_dummy = pd.DataFrame({'Scenario': scenario_names,\n",
        "                            'Sensor': 'DUMMY_SENSOR_UNDETECTED',\n",
        "                            'Impact': undetected_impact})\n",
        "    threshold = 1e-5\n",
        "    # threshold = 20\n",
        "    sensors = {}\n",
        "    for location in sensor_names:\n",
        "        position = chama.sensors.Stationary(location)\n",
        "        detector = chama.sensors.Point(threshold, sample_times)\n",
        "        stationary_pt_sensor = chama.sensors.Sensor(position, detector)\n",
        "        sensors[location] = stationary_pt_sensor\n",
        "\n",
        "    # Extract minimum detection time for each scenario-sensor pair\n",
        "    det_times = chama.impact.extract_detection_times(signal, sensors)\n",
        "    det_time_stats = chama.impact.detection_time_stats(det_times)\n",
        "    min_det_time = det_time_stats[['Scenario','Sensor','Min']]\n",
        "    min_det_time.rename(columns = {'Min':'Impact'}, inplace = True)\n",
        "\n",
        "    scenario_characteristics = pd.DataFrame({'Scenario': scenario_names,\n",
        "                                        'Undetected Impact': undetected_impact})\n",
        "    sensor_characteristics = pd.DataFrame({'Sensor': sensor_names,'Cost': 1})\n",
        "\n",
        "    cols = min_det_time.loc[:, \"Scenario\"].apply(node_to_id.get).values\n",
        "    rows = min_det_time.loc[:, \"Sensor\"].apply(node_to_id.get).values\n",
        "    values = min_det_time.loc[:, \"Impact\"]\n",
        "\n",
        "    dt = np.full((rows.max() + 1, cols.max() + 1), undetected_impact)\n",
        "    dt[rows, cols] = values\n",
        "    read_time = time.time()-time_s\n",
        "    return (read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6KU1WP2fZ3Ml"
      },
      "source": [
        "##Chama"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "QChnpP2PiQy0"
      },
      "outputs": [],
      "source": [
        "def run_by_chama(min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors):\n",
        "  time_s = time.time()\n",
        "  impactform = chama.optimize.ImpactFormulation()\n",
        "  result = impactform.solve(min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors)\n",
        "  wntr.graphics.plot_network(water_network, node_attribute=result['Sensors'])\n",
        "  chama_time = time.time()-time_s\n",
        "  return result, chama_time"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yxZ1-NkN8G9A"
      },
      "source": [
        "##GA"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "vVv2OJG_8Jk5"
      },
      "outputs": [],
      "source": [
        "log_node_iter = []\n",
        "log_population = []\n",
        "\n",
        "class Population:       # quần thể\n",
        "    def __init__(self, water_network, size_of_population, num_available_sensors, method, list_individuals) -> None:\n",
        "        self.water_network = water_network\n",
        "        self.scenario_names = water_network.node_name_list\n",
        "        self.sensor_names = water_network.junction_name_list\n",
        "        self.num_sensors = len(self.sensor_names)\n",
        "        self.size_of_population = size_of_population\n",
        "        self.num_available_sensors = num_available_sensors\n",
        "        self.method = method\n",
        "        self.list_individuals = list_individuals\n",
        "        self.best_individual = None\n",
        "        self.best_sensors = None\n",
        "        self.score = 0\n",
        "        self.parents = []\n",
        "\n",
        "\n",
        "    def choosed_sensors(self, individual):      # from idx -> list sensor\n",
        "        try:\n",
        "            _choosed = [self.sensor_names[i] for i in individual]\n",
        "        except:\n",
        "            print(self.sensor_names[i])\n",
        "            print(individual)\n",
        "            input()\n",
        "        return _choosed\n",
        "\n",
        "    # đột biến\n",
        "    def mutate(self, individual):\n",
        "        def _mutate():\n",
        "            new_sensor = random.choice(range(self.num_sensors))\n",
        "            while new_sensor in individual:\n",
        "                new_sensor = random.choice(range(self.num_sensors))\n",
        "            idx = np.random.randint(len(individual))\n",
        "            individual[idx] = new_sensor\n",
        "            individual.sort()\n",
        "            return individual\n",
        "\n",
        "        _new = _mutate()\n",
        "        return _new\n",
        "\n",
        "    def evaluate(self):\n",
        "        times = np.asarray([ self.method(min_det_time, self.choosed_sensors(individual)) for individual in self.list_individuals])\n",
        "        self.score = np.min(times)\n",
        "       \n",
        "        self.best_individual = self.list_individuals[times.tolist().index(self.score)]\n",
        "        \n",
        "        self.best_sensors = self.choosed_sensors(self.best_individual)\n",
        "        log_node_iter.append(self.best_sensors)\n",
        "        log_population.append(self.list_individuals)\n",
        "        self.parents.append(self.best_individual)\n",
        "        if False in (times[0] == times):\n",
        "            distances = np.max(times) - times\n",
        "        return times / np.sum(times)\n",
        "\n",
        "\n",
        "    def select(self, num_parents):\n",
        "        time_s = time.time()\n",
        "        fit = self.evaluate()\n",
        "#         print(\"Time eval: \", time.time() - time_s)\n",
        "        while len(self.parents) < num_parents:\n",
        "            idx = np.random.randint(0, self.size_of_population)\n",
        "            if fit[idx] > np.random.uniform(0, 1/self.size_of_population, size=(1,))[0]:\n",
        "                self.parents.append(self.list_individuals[idx])\n",
        "\n",
        "        self.parents = np.asarray(self.parents)\n",
        "\n",
        "    def crossover(self, p_cross=0.75):\n",
        "        def cross():\n",
        "            id1, id2 = np.random.choice(len(self.parents), size=2, replace=False)\n",
        "            parent1, parent2 = self.parents[id1], self.parents[id2]\n",
        "           # print(parent1)\n",
        "           # print(parent2)\n",
        "            gen_set=set(np.concatenate((parent1, parent2)))\n",
        "            gen_set=list(gen_set)\n",
        "            child = choice(gen_set, size=(self.num_available_sensors,), replace=False)\n",
        "            child.sort()\n",
        "            #print(child)\n",
        "            return child\n",
        "            \n",
        "        children = []\n",
        "        count = len(self.parents)\n",
        "        for _ in range(self.size_of_population):\n",
        "            if np.random.rand() > p_cross:\n",
        "                _tmp = random.choice(self.parents)\n",
        "                children.append(_tmp)\n",
        "                \n",
        "            else:\n",
        "                child = cross()\n",
        "                children.append(child)\n",
        "\n",
        "        # print(children)\n",
        "        return children\n",
        "\n",
        "    def next_population(self, p_cross=0.75, p_mutate=0.1):\n",
        "        _best_individual = self.best_individual\n",
        "        \n",
        "        _next = []\n",
        "        children = self.crossover(p_cross)\n",
        "        for child in children:\n",
        "            # print(child.selected_sensors)\n",
        "            if np.random.rand() < p_mutate:\n",
        "                # print(\"mutate\")\n",
        "                child_mutate = self.mutate(child)\n",
        "                # print(child_mutate)\n",
        "                if self.method(min_det_time, self.choosed_sensors(_best_individual))  >  self.method(min_det_time, self.choosed_sensors(child_mutate)):\n",
        "                    _next.append(child_mutate)\n",
        "                else:\n",
        "                    _next.append(_best_individual)\n",
        "                    _best_individual = child_mutate\n",
        "\n",
        "            else:\n",
        "                # print(\"un_mutate\")\n",
        "                # print(child)\n",
        "                if self.method(min_det_time, self.choosed_sensors(_best_individual))  >  self.method(min_det_time, self.choosed_sensors(child)):\n",
        "                    _next.append(child)\n",
        "                else:\n",
        "                    _next.append(_best_individual)\n",
        "                    _best_individual = child\n",
        "\n",
        "        if(self.best_individual not in _next):\n",
        "            _next[-1] = self.best_individual\n",
        "        return _next\n",
        "\n",
        "\n",
        "def init_population(water_network, size_of_population, num_available_sensors, method):\n",
        "    # khởi tạo 1 cá thể ngẫu nhiên\n",
        "    def init_individual(sensor_names, num_available_sensors):\n",
        "        \n",
        "        individual = np.random.choice(range(len(sensor_names)), size=num_available_sensors, replace=False)\n",
        "        individual.sort()\n",
        "        return individual\n",
        "\n",
        "    sensor_names = water_network.junction_name_list\n",
        "    population = []\n",
        "    for id in range(0, size_of_population):\n",
        "        population.append( init_individual(sensor_names, num_available_sensors) )\n",
        "    return Population(water_network, size_of_population, num_available_sensors, method, population )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "_mvHQhuJ7jaw"
      },
      "outputs": [],
      "source": [
        "def detection_time_GA(impact, selected_sensors):\n",
        "    sensors_id = [node_to_id.get(s) for s in selected_sensors]\n",
        "    return dt[sensors_id, :].min(axis=0).mean()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "KGIQcyw784rI"
      },
      "outputs": [],
      "source": [
        "def genetic_algorithm(\n",
        "    water_network,\n",
        "    size_of_population,\n",
        "    num_available_sensors,\n",
        "    method,\n",
        "    selectivity=0.15,\n",
        "    n_iter=100,\n",
        "    p_cross=0.75,\n",
        "    p_mut=0.1,\n",
        "    print_interval=100,\n",
        "    return_history=False,\n",
        "    verbose=False,\n",
        "):\n",
        "    t1 = time.time()\n",
        "    pop = init_population(water_network, size_of_population, num_available_sensors, method)\n",
        "    best_sensors = pop.best_sensors\n",
        "    score = float(\"inf\")\n",
        "    history = []\n",
        "    for i in range(n_iter):\n",
        "        pop.select(size_of_population * selectivity)\n",
        "        # print(pop.parents)\n",
        "        history.append(pop.score)\n",
        "        # if verbose:\n",
        "        #     print(f\"Generation {i}: {pop.score}\")\n",
        "        # elif i % print_interval == 0:\n",
        "        #     print(f\"Generation {i}: {pop.score}\")\n",
        "        if pop.score < score:\n",
        "            best_sensors = pop.best_sensors\n",
        "            score = pop.score\n",
        "        \n",
        "        children = pop.next_population(p_cross, p_mut)\n",
        "        \n",
        "        pop = Population(water_network, size_of_population, num_available_sensors, method, children)\n",
        "    if return_history:\n",
        "        return best_sensors, score, history\n",
        "    return best_sensors, score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "P_FZ6xe68679"
      },
      "outputs": [],
      "source": [
        "def run_ga():\n",
        "  log_node_iter = []\n",
        "  log_population = []\n",
        "\n",
        "  time_s = time.time()\n",
        "  best, score, history = genetic_algorithm(\n",
        "      water_network = water_network,\n",
        "      size_of_population=n_scenarios*20,\n",
        "      num_available_sensors=5,\n",
        "      method=detection_time_GA,\n",
        "      n_iter=100,\n",
        "      selectivity=0.15,\n",
        "      p_cross=0.75,\n",
        "      p_mut=0.1,\n",
        "      print_interval=5,\n",
        "      verbose=True,\n",
        "      return_history=True\n",
        "  )\n",
        "  GA_time = time.time() - time_s\n",
        "  #plt.plot(range(len(history)), history, color=\"blue\")\n",
        "  #wntr.graphics.plot_network(water_network, node_attribute=best)\n",
        "  #plt.show()\n",
        "\n",
        "  return GA_time, score, best\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ovu4TigxKxN9"
      },
      "source": [
        "##PSO"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "pj30q6gQK1Xl"
      },
      "outputs": [],
      "source": [
        "def run_pso():\n",
        "  options = {'c1': 0.5, 'c2': 0.5, 'w':0.9}\n",
        "  bounds = (-2000. * np.ones(n_scenarios), 2000. * np.ones(n_scenarios))\n",
        "  time_s = time.time()\n",
        "  # Call instance of PSO\n",
        "  optimizer = ps.single.GlobalBestPSO(n_particles = int(n_scenarios*2), dimensions=n_scenarios, options=options, bounds=bounds)\n",
        "\n",
        "  # Objective function\n",
        "  def detection_time_PSO(swarms):\n",
        "      sensor_ids = np.argpartition(swarms, -5, axis=1)[:, -5:]  # (n_particles, n_sensors)\n",
        "      output = dt[sensor_ids].min(axis=1).mean(axis=1)\n",
        "      return output\n",
        "\n",
        "  best_cost, best_pos = optimizer.optimize(detection_time_PSO, iters=1000)\n",
        "  PSO_time = time.time() - time_s\n",
        "  #plot_cost_history(optimizer.cost_history)\n",
        "  #plt.show()\n",
        "  # position\n",
        "  best_pos = np.argpartition(best_pos, -5)[-5:]\n",
        "  PSO_pos = [water_network.node_name_list[a] for a in best_pos]\n",
        "  # draw\n",
        "  wntr.graphics.plot_network(water_network, node_attribute=PSO_pos)\n",
        "  return  PSO_time, best_cost, PSO_pos\n",
        "  \n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3jLJLKxHIJg9"
      },
      "source": [
        "#test"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls"
      ],
      "metadata": {
        "id": "hroQZ3jZhMZd",
        "outputId": "2884bd17-1f8d-4c11-afa5-73abea82f0a9",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "rensors-distribution  report.log  sample_data\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qchu4c6aIRi5",
        "outputId": "5d178c3f-973a-49e9-8041-5166c1c78bb7"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'Controls': 2,\n",
              " 'Curves': 1,\n",
              " 'Links': 13,\n",
              " 'Nodes': 11,\n",
              " 'Patterns': 1,\n",
              " 'Sources': 0}"
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ],
      "source": [
        "path = \"rensors-distribution/data/Net1.inp\"\n",
        "net_info = read_network(path)\n",
        "net_info"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_1_GMR0bIlc5",
        "outputId": "f4436c4b-5706-45df-b72b-74b86182f9f4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "read time 0.36299943923950195\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/pandas/core/frame.py:5047: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  errors=errors,\n"
          ]
        }
      ],
      "source": [
        "read_time, water_network, min_det_time, dt, node_to_id, n_scenarios, sensor_characteristics, scenario_characteristics = input_network(path)\n",
        "print('read time', read_time)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 530
        },
        "id": "v5EfQTuccp2-",
        "outputId": "5c908989-3c9d-4784-d33d-247d5cc41410"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "GA time 4.577558994293213\n",
            "GA result 7527.272727272727\n"
          ]
        }
      ],
      "source": [
        "GA_time, GA_score, GA_best = run_ga()\n",
        "print('GA time',GA_time)\n",
        "print('GA result', GA_score)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 850
        },
        "id": "Wb4y1lOVctuh",
        "outputId": "2f366be9-f1c0-4bd7-8adc-7378f14b6776"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2022-04-29 04:15:33,964 - pyswarms.single.global_best - INFO - Optimize for 1000 iters with {'c1': 0.5, 'c2': 0.5, 'w': 0.9}\n",
            "pyswarms.single.global_best: 100%|██████████|1000/1000, best_cost=1.96e+3\n",
            "2022-04-29 04:15:37,694 - pyswarms.single.global_best - INFO - Optimization finished | best cost: 1963.6363636363637, best pos: [-1.082e-01  8.939e+02  1.472e+02 -1.543e+02 -5.329e+02 -1.396e+03  8.094e+02 -1.928e+03 -1.864e+03  1.516e+03  1.977e+03]\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x576 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "PSO time 3.751065969467163\n",
            "PSO result 1963.6363636363637\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "PSO_time, PSO_score, PSO_pos = run_pso()\n",
        "print('PSO time',PSO_time)\n",
        "print('PSO result', PSO_score)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "metadata": {
        "id": "VNCfFYLRcw-Q",
        "outputId": "560c2ce5-fbe0-4a90-ca81-fb80397d8f97",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 283
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Chama time 0.13740158081054688\n",
            "Chama result 1963.6363636363637\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "limit_sensors = 5\n",
        "chama_result, chama_time = run_by_chama(min_det_time, sensor_characteristics, scenario_characteristics, limit_sensors)\n",
        "print('Chama time',chama_time)\n",
        "print('Chama result', chama_result['Objective'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "id": "mFZT1g_Xc2ej",
        "outputId": "0931822e-d269-4f3c-a8bc-1dca4283704c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 162
        }
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                  Net sensors Nodes Links  \\\n",
              "0  rensors-distribution/data/Net1.inp       5    11    13   \n",
              "\n",
              "             read_time           chama_time         chama_score  \\\n",
              "0  0.36299943923950195  0.13740158081054688  1963.6363636363637   \n",
              "\n",
              "             GA_time           GA_score           PSO_time           PSO_score  \n",
              "0  4.577558994293213  7527.272727272727  3.751065969467163  1963.6363636363637  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-37101bde-805a-4a7d-ac7f-ae460a720d8a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Net</th>\n",
              "      <th>sensors</th>\n",
              "      <th>Nodes</th>\n",
              "      <th>Links</th>\n",
              "      <th>read_time</th>\n",
              "      <th>chama_time</th>\n",
              "      <th>chama_score</th>\n",
              "      <th>GA_time</th>\n",
              "      <th>GA_score</th>\n",
              "      <th>PSO_time</th>\n",
              "      <th>PSO_score</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>rensors-distribution/data/Net1.inp</td>\n",
              "      <td>5</td>\n",
              "      <td>11</td>\n",
              "      <td>13</td>\n",
              "      <td>0.36299943923950195</td>\n",
              "      <td>0.13740158081054688</td>\n",
              "      <td>1963.6363636363637</td>\n",
              "      <td>4.577558994293213</td>\n",
              "      <td>7527.272727272727</td>\n",
              "      <td>3.751065969467163</td>\n",
              "      <td>1963.6363636363637</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-37101bde-805a-4a7d-ac7f-ae460a720d8a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-37101bde-805a-4a7d-ac7f-ae460a720d8a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-37101bde-805a-4a7d-ac7f-ae460a720d8a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 33
        }
      ],
      "source": [
        "data = np.array([[path, limit_sensors, net_info['Nodes'], net_info['Links'],\n",
        "                  read_time, chama_time, chama_result['Objective'],\n",
        "                  GA_time, GA_score, PSO_time, PSO_score]])\n",
        "df = pd.DataFrame(data, columns=['Net', 'sensors', 'Nodes', 'Links','read_time', 'chama_time', 'chama_score',\n",
        "                                 'GA_time', 'GA_score', 'PSO_time', 'PSO_score'])\n",
        "df"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "sensor opt compare.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}