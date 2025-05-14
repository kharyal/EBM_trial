import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Energy Based Model (EBM) for Concept Learning
takes a set of 5 points and a concept as input, outputs the energy
'''

class ConceptEBM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super(ConceptEBM, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.context_net = nn.TransformerEncoder(transformer_layer, num_layers=4)

        self.g_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, observer, concept):
        # x: (batch_size, num_points, input_dim)
        x[..., :2] = x[..., :2] - observer.unsqueeze(1)
        batch_size, num_points, _ = x.size()
        concept = concept.unsqueeze(1).expand(batch_size, num_points, -1)
        x = torch.cat((x, concept), dim=-1)
        x = self.feature_net(x)
        x = self.context_net(x).mean(dim=1)  # (batch_size, hidden_dim)
        x = self.g_net(x)
        return x
        

class ConceptEBMMLP(nn.Module):
    '''
        f( \sum_{i,j} g(x_i, x_j, c), c)
    '''
    def __init__(self, input_dim=13, hidden_dim=128):
        super(ConceptEBMMLP, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )
        self.g_net = nn.Sequential(
            nn.Linear(hidden_dim + 10, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, observer, concept):
        # x: (batch_size, num_points, input_dim)
        x[..., :2] = x[..., :2] - observer.unsqueeze(1)
        batch_size, num_points, _ = x.size()
        concept_ = concept.unsqueeze(1).expand(batch_size, num_points, -1)
        # \sum_{i,j} g(x_i, x_j)
        x = torch.cat((x, concept_), dim=-1)
        x_ = self.feature_net(x)
        attn = self.attention_net(x)
        x = x_ * attn
        
        x = x.view(batch_size, num_points, -1).mean(dim=1)

        x = torch.cat((x, concept), dim=-1)
        x = self.g_net(x)
        return x