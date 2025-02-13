from model.GCNModel import GCNModel
from model.MLP import MLP
from model.TripletLoss import TripletLoss
import torch.nn as nn
import torch
class GCNMLPTripletLossModel(nn.Module):
    def __init__(self, gcn_in_channels, gcn_out_channels, mlp_hidden_dim, mlp_output_dim):
        super(GCNMLPTripletLossModel, self).__init__()
        
        # Mô hình GCN
        self.gcn = GCNModel(gcn_in_channels, gcn_out_channels)
        
        # MLP
        self.mlp = MLP(gcn_out_channels, mlp_hidden_dim, mlp_output_dim)
        
        # Triplet Loss
        self.triplet_loss = TripletLoss(margin=1.0)

    def forward(self, data, anchor_idx, positive_idx, negative_idx):
        # Trích xuất đặc trưng từ GCN
        gcn_output = self.gcn(data)
        
        # Lấy các điểm đặc trưng cho anchor, positive, và negative từ GCN output
        anchor_output = gcn_output[anchor_idx]
        positive_output = gcn_output[positive_idx]
        negative_output = gcn_output[negative_idx]
        
        # Đưa đặc trưng vào MLP
        anchor_output = self.mlp(anchor_output)
        positive_output = self.mlp(positive_output)
        negative_output = self.mlp(negative_output)
        
        # Tính Triplet Loss
        loss = self.triplet_loss(anchor_output, positive_output, negative_output)
        
        return loss
