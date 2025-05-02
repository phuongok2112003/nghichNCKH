import torch

graphs = torch.load('graphs.pt',weights_only=False)


# for graph in graphs:

#    print(graph.y[0].item())
print(graphs[0])
