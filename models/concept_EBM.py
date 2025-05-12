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

    def forward(self, x, concept):
        # x: (batch_size, num_points, input_dim)
        batch_size, num_points, _ = x.size()
        concept = concept.unsqueeze(1).expand(batch_size, num_points, -1)
        x = torch.cat((x, concept), dim=-1)
        x = self.feature_net(x)
        x = self.context_net(x).mean(dim=1)  # (batch_size, hidden_dim)
        x = self.g_net(x)
        return x
        