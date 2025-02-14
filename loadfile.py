import torch

graphs = torch.load('graphs.pt')


for graph in graphs:

   print(graph.edge_attr.shape)
