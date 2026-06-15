import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

# Load Schema Dataset
def load_dict(path):
    d = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                d[parts[1]] = int(parts[0])
    return d

ent_dict = load_dict('kge/data/schema_dataset/entities.dict')
rel_dict = load_dict('kge/data/schema_dataset/relations.dict')

triples = []
with open('kge/data/schema_dataset/train.txt', 'r') as f:
    for line in f:
        h, r, t = line.strip().split('\t')
        triples.append((ent_dict[h], rel_dict[r], ent_dict[t]))

# Simple Q2B model for training
class SimpleQ2B(nn.Module):
    def __init__(self, nent, nrel, dim):
        super().__init__()
        self.entity_embedding = nn.Embedding(nent, dim)
        self.relation_embedding = nn.Embedding(nrel, dim)
        self.offset_embedding = nn.Embedding(nrel, dim)
        
        # Initialize
        nn.init.uniform_(self.entity_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.offset_embedding.weight, -0.1, 0.1)

    def forward(self, h, r):
        # Center projection: c = h + r
        c = self.entity_embedding(h) + self.relation_embedding(r)
        # Offset projection: o = h_o + Softplus(r_o)
        # For 1p queries, h_o is 0
        o = F.softplus(self.offset_embedding(r))
        return c, o

def train_schema():
    dim = 64
    model = SimpleQ2B(len(ent_dict), len(rel_dict), dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # We want to minimize the distance between (h+r) and t
    # And keep the offset small but sufficient
    for epoch in range(1000):
        total_loss = 0
        for h, r, t in triples:
            optimizer.zero_grad()
            
            h_t = torch.tensor([h])
            r_t = torch.tensor([r])
            t_t = torch.tensor([t])
            
            c, o = model(h_t, r_t)
            target = model.entity_embedding(t_t)
            
            # Distance loss (Center must be close to target)
            dist = torch.norm(c - target, p=2)
            
            # Box loss (Target should be inside the box)
            # This is a simplified version: ensure target is within offset
            # But for training, just minimizing distance is usually enough for a starting point
            loss = dist + 0.1 * torch.norm(o, p=1)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # Save checkpoint
    checkpoint = {
        'model_state_dict': {
            'entity_embedding': model.entity_embedding.weight.data,
            'relation_embedding': model.relation_embedding.weight.data,
            'offset_embedding': model.offset_embedding.weight.data
        }
    }
    torch.save(checkpoint, 'kge/models/q2b_model/checkpoint_schema.pth')
    print("Checkpoint saved to kge/models/q2b_model/checkpoint_schema.pth")

if __name__ == "__main__":
    train_schema()
