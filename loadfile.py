import torch

graphs = torch.load('graphs.pt')


# for graph in graphs:

#    print(graph.y[0].item())
print(graphs[0].edge_attr)
