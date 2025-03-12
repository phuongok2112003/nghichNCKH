import networkx as nx
from pyvis.network import Network

# Đọc đồ thị từ file GraphML
file_path = "output/0_0.c/export.xml"  # Đổi đường dẫn nếu cần
G = nx.read_graphml(file_path)

# Tạo đồ thị tương tác với Pyvis
net = Network(notebook=True, directed=True)  # directed=True để hiển thị hướng cạnh

# Thêm node và thông tin thuộc tính
for node, data in G.nodes(data=True):
    label = f"Node {node}"
    tooltip = ", ".join(f"{key}: {value}" for key, value in data.items())  # Hiển thị cả key và value
    net.add_node(node, label=label, title=tooltip)



# Thêm cạnh và hiển thị thuộc tính
for source, target, data in G.edges(data=True):
    tooltip = ", ".join(f"{key}: {value}" for key, value in data.items())  # Hiển thị cả key và value
    net.add_edge(source, target, title=tooltip)



# Lưu và mở file HTML
net.show("graph.html")
