import wntr


def read_network(path):
    water_network = wntr.network.WaterNetworkModel(path)
    info = water_network.describe(level=0)
    return (path, info['Nodes'], info['Links'])
