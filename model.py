# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class EGNNLayer(nn.Module):
    def __init__(self, hidden_nf, edge_attr_dim=2, act_fn=nn.GELU()):
        super(EGNNLayer, self).__init__()
        self.hidden_nf = hidden_nf
        
        
        edge_in_dim = hidden_nf * 2 + 1 + edge_attr_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf), act_fn
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2, hidden_nf), act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )

    def forward(self, h, edge_index, coord, edge_attr):
        row, col = edge_index
        
        
        radial = torch.sum((coord[row] - coord[col])**2, dim=1).unsqueeze(1)
        
        edge_feat = torch.cat([h[row], h[col], radial, edge_attr], dim=1)
        m_ij = self.edge_mlp(edge_feat)
        
        m_i = torch.zeros(h.size(0), m_ij.size(1), dtype=h.dtype, device=h.device)
        m_i.scatter_add_(0, row.unsqueeze(1).expand_as(m_ij), m_ij)
        
        node_feat = torch.cat([h, m_i], dim=1)
        h_out = h + self.node_mlp(node_feat) 
        
        return h_out


class StructuralEGNN(nn.Module):
    def __init__(self, hidden_nf=512, out_node_nf=1, device='cpu', 
                 egnn_layers=2, return_features=False, dropout_rate=0.3, 
                 input_seq_dim=2560, transformer_layers=2, transformer_dim=1024, **kwargs):
        super(StructuralEGNN, self).__init__()
        
        self.device = device
        self.hidden_nf = hidden_nf
        self.return_features = return_features
        self.transformer_dim = transformer_dim
        
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_seq_dim, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim)
        )
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim, 
            nhead=8,                     
            dim_feedforward=self.transformer_dim * 4,
            dropout=dropout_rate,
            batch_first=True,             
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        
        self.trans_to_egnn = nn.Sequential(
            nn.Linear(self.transformer_dim, hidden_nf), 
            nn.LayerNorm(hidden_nf)
        )
        
        
        self.egnn_layers_list = nn.ModuleList([
            EGNNLayer(hidden_nf=hidden_nf, edge_attr_dim=2) 
            for _ in range(egnn_layers)
        ])
        
        
        classifier_in_dim = input_seq_dim + hidden_nf 
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(128, out_node_nf) 
        )

    def forward(self, seq_features, structural_input):
        mask = structural_input['mask'] 
        batch_size, max_seq_len = mask.shape
        
       
        x = self.input_proj(seq_features) 
        
       
        padding_mask = ~mask 
        h_seq_1024 = self.transformer_encoder(x, src_key_padding_mask=padding_mask) 
        
        h_seq_256 = self.trans_to_egnn(h_seq_1024) 
        
       
        h = h_seq_256.view(batch_size * max_seq_len, -1) 

        flat_seq_features = seq_features.view(batch_size * max_seq_len, -1)
        
        coords = structural_input['coordinates']
        edges = structural_input['edges']
        edge_attr = structural_input['edge_attr']

        for egnn_layer in self.egnn_layers_list:
            h = egnn_layer(h, edges, coords, edge_attr)
            
        if self.return_features:
            return h
            
        
        concat_features = torch.cat([flat_seq_features, h], dim=-1)
        logits_flat = self.classifier(concat_features)
        
        logits = logits_flat.view(batch_size, max_seq_len, 1)
        out = torch.full((batch_size, max_seq_len, 1), -1e6, device=self.device)
        out[mask] = logits[mask]
        
        return out

def get_model(**kwargs):
    return StructuralEGNN(**kwargs)