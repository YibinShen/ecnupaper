import copy
import torch
import torch.nn as nn
from transformers.activations import gelu_new as gelu_bert


class GCN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x, adj):
        x = torch.relu(torch.matmul(adj, self.gc1(x)))
        x = torch.relu(torch.matmul(adj, self.gc2(x)))
        return x

class Graph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout_expand1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_out1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_expand2 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_out2 = nn.Dropout(config.hidden_dropout_prob)
        self.gcn1 = GCN(self.hidden_size, self.intermediate_size)
        self.gcn2 = GCN(self.hidden_size, self.intermediate_size)
        self.lin_expand1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.lin_collapse1 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.lin_expand2 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.lin_collapse2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.gate_weight = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.norm1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
    
    def normalize(self, A, symmetric=True):
        d = A.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else :
            D = torch.diag(torch.pow(d, -1))
            return D.mm(A)
       
    def b_normal(self, adj, symmetric=True):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i], symmetric)
        return adj

    def forward(self, nodes, graphs):
        adjs = graphs.float()
        adjs_list = [adjs[:,0,:]] + [adjs[:,1,:]]
        adjs_list = [self.b_normal(x) for x in adjs_list]
        g_feature1 = self.gcn1(nodes, adjs_list[0])
        g_feature2 = self.gcn2(nodes, adjs_list[1])
        g_feature1 = self.lin_collapse1(self.dropout_expand1(gelu_bert(self.lin_expand1(g_feature1))))
        g_feature1 = self.norm1(nodes + self.dropout_out1(g_feature1))
        g_feature2 = self.lin_collapse2(self.dropout_expand2(gelu_bert(self.lin_expand2(g_feature2))))
        g_feature2 = self.norm2(nodes + self.dropout_out2(g_feature2))

        gate = torch.cat((g_feature1, g_feature2, g_feature1+g_feature2, g_feature1-g_feature2), dim=2)
        gate = torch.sigmoid(self.gate_weight(gate))
        g_feature = gate * g_feature1 + (1-gate) * g_feature2

        return g_feature