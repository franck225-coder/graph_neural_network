import torch
from torch_geometric.data import Data

# edge_index: is a 2D array where the second dimension consists of 2
# subarrays representing the Origin and Destination nodes
# (eg. from node 1 to node 2, from node 0 to node 4, from node 0 to node 1, and from node 1 to node 3)
edge_index = torch.tensor([[1,0,0,1],[2,4,1,3]], dtype=torch.long)

# x: the value attributes of the three nodes
x = torch.tensor([[1],[0],[0]], dtype=torch.float)

# Data: constructs the graph data structure when you provide the x attributes and edge_index
data = Data(x=x, edge_index=edge_index)

# The number of nodes in the graph
print(data.num_nodes)

# The number of edges
print(data.num_edges)

# Number of attributes
print(data.num_node_features)

# If the graph contains any isolated nodes(true or flase)
print(data.contains_isolated_nodes())

