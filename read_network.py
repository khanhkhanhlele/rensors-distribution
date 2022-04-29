import wntr


def read_network(path):
    water_network = wntr.network.WaterNetworkModel(path)
    print(water_network.describe(level=0))
    return water_network.describe(level=0)
