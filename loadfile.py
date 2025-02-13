import torch

graphs = torch.load('graphs.pt')

print(graphs[0].edge_index)
