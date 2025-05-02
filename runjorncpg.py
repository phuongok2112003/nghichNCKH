
import torch
import xml.etree.ElementTree as ET
from torch_geometric.data import Data


import subprocess
import os
import networkx as nx
from torch_geometric.data import Data

def run_joern_analysis(c_file_path, output_dir):
    
    
    file_name=c_file_path.split('/')[-1].split('\\')[-1]
    export_file = os.path.join(output_dir, file_name)
    print(f"Đang xuất đồ thị từ CPG vào {export_file}...")
    if os.path.isdir(f"{output_dir}/{file_name}"):
        print(f"Thư mục {output_dir}/{file_name} da tồn tại.")
        return None
    
    print(f"Đang phân tích mã nguồn từ {c_file_path}...")
    parse_command = f"joern-parse {c_file_path}"
    subprocess.run(parse_command, shell=True, check=True)

   
    export_command = f"joern-export --repr=all --format=graphml --out {output_dir}/{file_name}"
    subprocess.run(export_command, shell=True, check=True)

   
    print(f"Đang đọc đồ thị từ {export_file}...")
    graph = nx.read_graphml(f"{output_dir}/{file_name}/export.xml")  

   
    node_features = []
    node_mapping = {}  

    for idx, node in enumerate(graph.nodes()):
        degree = graph.degree(node)  
        node_features.append([degree])
        node_mapping[node] = idx  

   
    x = torch.tensor(node_features, dtype=torch.float)

    
    edges = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    
    data = Data(x=x, edge_index=edge_index)

   
    return data

def process_directory(directory, output_dir):
   
    c_files = [f for f in os.listdir(directory) if f.endswith(".c")]
    os.makedirs(output_dir, exist_ok=True)
 
    for c_file in c_files:
        c_file_path = os.path.join(directory, c_file)
        print(f"Đang xử lý tệp: {c_file}")

        
        data = run_joern_analysis(c_file_path, output_dir)

       
        print(data)


c_directory ="I:\\raw_code"
output_dir = "G:/output"


process_directory(c_directory, output_dir)
