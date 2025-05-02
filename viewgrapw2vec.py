import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from pyvis.network import Network

graphs = torch.load('graphs.pt',weights_only=False)

data = graphs[0]
# Chuyển sang NetworkX graph
G = to_networkx(data, to_undirected=True)  # hoặc to_undirected=False nếu bạn muốn giữ hướng cạnh

# Tạo đồ thị tương tác bằng Pyvis
net = Network(notebook=True, directed=True)

# Thêm node: in toàn bộ đặc trưng `x` không làm tròn
for node in G.nodes():
    feature_vec = data.x[node].tolist()
    feature_str = str(feature_vec)  # không làm tròn
    net.add_node(
        node,
        label=f"Node {node}",
        title=f"x: {feature_str}",
    )

# Thêm cạnh: in toàn bộ `edge_attr` không làm tròn
edge_index = data.edge_index
edge_attr = data.edge_attr

for i in range(edge_index.size(1)):
    src = edge_index[0, i].item()
    tgt = edge_index[1, i].item()
    attr_vec = edge_attr[i].tolist()
    attr_str = str(attr_vec)  # không làm tròn
    net.add_edge(
        src,
        tgt,
        title=f"edge_attr: {attr_str}"
    )

# Xuất file HTML
net.show("graph_with_full_features.html")