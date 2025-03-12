
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,GlobalAttention
from torch_geometric.data import Data, DataLoader
import numpy as np
from torch.nn import Linear
import joblib
import torch
import xml.etree.ElementTree as ET
from torch_geometric.data import Data
from gensim.models import Word2Vec
import numpy as np

import subprocess
import os
import networkx as nx
from torch_geometric.data import Data

def run_joern_analysis(c_file_path, output_dir):
    # Đảm bảo thư mục output tồn tại
    
    file_name=c_file_path.split('/')[-1].split('\\')[-1]
    export_file = os.path.join(output_dir, file_name)
    print(f"Đang xuất đồ thị từ CPG vào {export_file}...")
    if os.path.isdir(f"{output_dir}/{file_name}"):
        print(f"Thư mục {output_dir}/{file_name} da tồn tại.")
        return None
    # 1. Phân tích mã nguồn và tạo CPG
    print(f"Đang phân tích mã nguồn từ {c_file_path}...")
    parse_command = f"joern-parse {c_file_path}"
    subprocess.run(parse_command, shell=True, check=True)

   
    export_command = f"joern-export --repr=all --format=graphml --out {output_dir}/{file_name}"
    subprocess.run(export_command, shell=True, check=True)

    # 3. Đọc đồ thị XML và tạo đối tượng NetworkX
    print(f"Đang đọc đồ thị từ {export_file}...")
    graph = nx.read_graphml(f"{output_dir}/{file_name}/export.xml")  # Mặc dù chúng ta xuất ra XML, nhưng NetworkX hỗ trợ đọc XML dạng GraphML

    # 4. Tạo các đặc trưng cho các nút (nếu cần)
    node_features = []
    node_mapping = {}  # Lưu trữ mapping từ tên nút sang chỉ số

    for idx, node in enumerate(graph.nodes()):
        degree = graph.degree(node)  # Đặc trưng độ của nút
        node_features.append([degree])
        node_mapping[node] = idx  # Đánh chỉ số cho các nút

    # Chuyển đổi các đặc trưng thành tensor PyTorch
    x = torch.tensor(node_features, dtype=torch.float)

    # 5. Tạo các cạnh đồ thị (dùng chỉ số của các nút)
    edges = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 6. Tạo đối tượng Data trong PyTorch Geometric
    data = Data(x=x, edge_index=edge_index)

    # Trả về đối tượng đồ thị PyTorch Geometric
    return data

def process_directory(directory, output_dir):
    # Lọc các tệp .c trong thư mục
    # c_files = [f for f in os.listdir(directory) if f.endswith(".c")]
    os.makedirs(output_dir, exist_ok=True)
    # Duyệt qua tất cả các tệp .c và phân tích
    for c_file in directory:
        # c_file_path = os.path.join(directory, c_file)
        print(f"Đang xử lý tệp: {c_file}")

        # Phân tích và lấy đồ thị từ mã nguồn C
        data = run_joern_analysis(c_file, output_dir)

        # In thông tin của đối tượng Data (có thể thay đổi tuỳ ý)
        print(data)

# Đường dẫn tới thư mục chứa mã nguồn C và thư mục xuất kết quả
c_directory =["F:\\NCKH\\chrome_debian\\raw_code\\0_0.c","F:\\NCKH\\chrome_debian\\raw_code\\2_1.c","F:\\NCKH\\chrome_debian\\raw_code\\3_1.c","F:\\NCKH\\chrome_debian\\raw_code\\3_0.c"]
output_dir = "G:/output"  # Thư mục xuất kết quả

# Xử lý tất cả các tệp .c trong thư mục
process_directory(c_directory, output_dir)




# Namespace cho GraphML
NAMESPACE = {"ns": "http://graphml.graphdrawing.org/xmlns"}

# Hàm đọc XML và chuyển thành đồ thị PyG
def load_graph_from_folder(folder_path):
    file_path = os.path.join(folder_path, "export.xml")

    # Đọc file XML
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Dictionary lưu thông tin node & mapping
    node_features = {}
    node_mapping = {}

    # Lấy danh sách node
    for idx, node in enumerate(root.findall("ns:graph/ns:node", NAMESPACE)):
        node_id = node.get("id")
        node_data = {}

        # Lấy thuộc tính của node
        for data in node.findall("ns:data", NAMESPACE):
            key = data.get("key")
            value = data.text if data.text else "UNKNOWN"
            node_data[key] = value

        # Lưu node vào dictionary
        node_features[node_id] = node_data
        node_mapping[node_id] = idx  # Đánh số lại cho node

    # Xử lý cạnh (edges) & thuộc tính cạnh
    edges = []
    edge_features = []

    for edge in root.findall("ns:graph/ns:edge", NAMESPACE):
        src = edge.get("source")
        tgt = edge.get("target")
        edge_data = {}

        # Lấy thuộc tính của cạnh
        for data in edge.findall("ns:data", NAMESPACE):
            key = data.get("key")
            value = data.text if data.text else "UNKNOWN"
            edge_data[key] = value

        if src in node_mapping and tgt in node_mapping:
            edges.append((node_mapping[src], node_mapping[tgt]))
            edge_features.append(edge_data)

    # Chuyển danh sách edges thành tensor
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        print(f"Lỗi: Không có cạnh trong {folder_path}")
        return None  # Bỏ qua nếu không có cạnh

    # Sử dụng Word2Vec để nhúng đặc trưng của node
    node_sentences = [list(data.values()) for data in node_features.values()]
    node_model = Word2Vec(sentences=node_sentences, vector_size=100, min_count=1, workers=4)

    x = []
    for data in node_features.values():
        words = list(data.values())
        vectors = [node_model.wv[word] for word in words if word in node_model.wv]
        x.append(np.mean(vectors, axis=0) if vectors else np.zeros(100))  # 100 chiều vector

    x = torch.tensor(x, dtype=torch.float)

    # Sử dụng Word2Vec để nhúng đặc trưng của cạnh
    edge_sentences = [list(data.values()) for data in edge_features]
    edge_model = Word2Vec(sentences=edge_sentences, vector_size=50, min_count=1, workers=4)  # Embedding cạnh 50 chiều

    edge_attr = []
    for data in edge_features:
        words = list(data.values())
        vectors = [edge_model.wv[word] for word in words if word in edge_model.wv]
        edge_attr.append(np.mean(vectors, axis=0) if vectors else np.zeros(50))  # 50 chiều vector

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Tạo đối tượng đồ thị PyG
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

# Hàm tải toàn bộ đồ thị
def load_all_graphs(base_path):
    graphs = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print(f"Đang xử lý thư mục: {folder_path}")
            graph = load_graph_from_folder(folder_path)
            if graph:
                # # Gắn nhãn dựa trên tên thư mục
                # try:
                #     label = int(folder.split('_')[1].split('.')[0])
                #     graph.y = torch.tensor([label], dtype=torch.long)  # Thêm nhãn
                # except (IndexError, ValueError):
                #     graph.y = torch.tensor([0], dtype=torch.long)  # Nhãn mặc định nếu lỗi
                graphs.append(graph)
    print(f"Tổng số đồ thị đã xử lý: {len(graphs)}")
    return graphs

# Đường dẫn thư mục chứa dữ liệu
base_path = "G:/output"

# Tải toàn bộ đồ thị
graphs = load_all_graphs(base_path)

# Kiểm tra dữ liệu đầu tiên
if graphs:
    print("Dữ liệu đồ thị đầu tiên:")
    print("Node features:", graphs[0].x.shape)
    print("Edge features:", graphs[0].edge_attr.shape)

# Lưu dữ liệu
# torch.save(graphs, 'graphs_process.pt')
# print("Đã lưu dữ liệu vào 'graphs.pt'")




# === 1. Define the Model ===
class NodeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.softmax(self.conv1(x, edge_index), dim=1)

        x = self.conv2(x, edge_index)
        return x

class EdgeCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, edge_attr):
       
        if edge_attr is None:
            print("None rồi ......................")
            return None

        if edge_attr.dim() == 2:
            edge_attr = edge_attr.permute(1, 0).unsqueeze(0)

        edge_attr = F.softmax(self.conv1(edge_attr), dim=1)

        edge_attr = self.conv2(edge_attr)
        
        return edge_attr.squeeze(0).permute(1, 0)

class GraphModel(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, node_output_dim, 
                 edge_input_dim, edge_output_dim, final_dim):
        super(GraphModel, self).__init__()
        self.node_gcn = NodeGCN(node_input_dim, node_hidden_dim, node_output_dim)
        self.edge_cnn = EdgeCNN(edge_input_dim, edge_output_dim)

        self.att = GlobalAttention(gate_nn=Linear(node_output_dim, 1))  # Thêm Attention Pooling

        graph_feature_dim = node_output_dim + edge_output_dim  # Tổng số chiều của đặc trưng graph
        self.fc = nn.Linear(graph_feature_dim, final_dim)

    def forward(self, data):
        node_features = self.node_gcn(data.x, data.edge_index)
        edge_features = self.edge_cnn(data.edge_attr) if data.edge_attr is not None else None

        # **Dùng Attention Pooling thay vì Mean/Max**
        node_features = self.att(node_features, data.batch)

        if edge_features is not None:
            edge_batch = data.batch[data.edge_index[0]]  
            edge_features = global_mean_pool(edge_features, edge_batch)
            graph_features = torch.cat([node_features, edge_features], dim=-1)
        else:
            graph_features = node_features

        out = self.fc(graph_features)
        return out, graph_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# graphs = torch.load('graphs_process.pt')

model = GraphModel(node_input_dim=100, node_hidden_dim=64, node_output_dim=32,
                   edge_input_dim=50, edge_output_dim=16, final_dim=2).to(device)
model.eval()
def extract_features(graphs):
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    features =[]
    print(len(loader))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, graph_features = model(data)
            features.append(graph_features.cpu().numpy())
       

    X = np.vstack(features)
 
    return X

X_test = extract_features(graphs)

clf_loaded = joblib.load("random_forest_model.pkl")

# Dự đoán trên tập test
y_pred = clf_loaded.predict(X_test)

print("Du doan ta duoc",y_pred)