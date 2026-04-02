
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionModel(nn.Module):
    def __init__(self, d_molecule, d_text, d_model):
        super(AttentionFusionModel, self).__init__()
        self.d_molecule = d_molecule
        self.d_text = d_text
        self.d_model = d_model
        self.scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        self.W_Q = nn.Linear(d_text, d_model)
        self.W_K = nn.Linear(d_text, d_model)
        self.W_V = nn.Linear(d_molecule, d_model)

    def forward(self, molecule_embedding, text_embedding):
        Q = self.W_Q(text_embedding)
        K = self.W_K(text_embedding)
        V = self.W_V(molecule_embedding)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        print(f"Attention Weights Shape: {attention_weights.shape}")
        print(f"Attention Weights: {attention_weights}")

        output = torch.matmul(attention_weights, V)
        return output

# Simulate Inference Case: 1 Text, 1 Molecule
d_mol = 10
d_text = 12
d_model = 8

model = AttentionFusionModel(d_mol, d_text, d_model)

mol_input = torch.randn(1, d_mol)
text_input1 = torch.randn(1, d_text)
text_input2 = torch.randn(1, d_text) # Different text

print("--- Case 1: Text A ---")
out1 = model(mol_input, text_input1)
print(f"Output 1: {out1}")

print("\n--- Case 2: Text B (Different) ---")
out2 = model(mol_input, text_input2)
print(f"Output 2: {out2}")

print(f"\nAre outputs identical? {torch.allclose(out1, out2)}")
