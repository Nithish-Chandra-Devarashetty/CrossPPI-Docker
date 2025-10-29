import torch
import torch.nn as nn
import os
from src.protein_processor import ProteinInference
from src.model import Protein_feature_extraction, cross_attention

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
hidden_dim = 128

# -------------------------------
# PPI Model Definition
# -------------------------------
class PPI(nn.Module):
    def __init__(self):
        super(PPI, self).__init__()
        self.ligand_graph_model = Protein_feature_extraction(hidden_dim)
        self.receptor_graph_model = Protein_feature_extraction(hidden_dim)
        self.cross_attention = cross_attention(hidden_dim)

        self.line1 = nn.Linear(hidden_dim * 2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

        self.ligand1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.receptor1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ligand2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.receptor2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, ligand_batch, receptor_batch):
        ligand_out_seq, ligand_out_graph, ligand_mask_seq, ligand_mask_graph, ligand_seq_final, ligand_graph_final = self.ligand_graph_model(ligand_batch, device)
        receptor_out_seq, receptor_out_graph, receptor_mask_seq, receptor_mask_graph, receptor_seq_final, receptor_graph_final = self.receptor_graph_model(receptor_batch, device)

        context_layer, _ = self.cross_attention(
            [ligand_out_seq, ligand_out_graph, receptor_out_seq, receptor_out_graph],
            [ligand_mask_seq, ligand_mask_graph, receptor_mask_seq, receptor_mask_graph],
            device
        )

        out_ligand = context_layer[-1][0]
        out_receptor = context_layer[-1][1]

        ligand_mask_combined = torch.cat((ligand_mask_seq, ligand_mask_graph), dim=1)
        receptor_mask_combined = torch.cat((receptor_mask_seq, receptor_mask_graph), dim=1)

        ligand_cross_seq = ((out_ligand * ligand_mask_combined.unsqueeze(dim=2)).mean(dim=1) + ligand_seq_final) / 2
        ligand_cross_stru = ((out_ligand * ligand_mask_combined.unsqueeze(dim=2)).mean(dim=1) + ligand_graph_final) / 2
        ligand_cross = (ligand_cross_seq + ligand_cross_stru) / 2
        ligand_cross = self.ligand2(self.dropout(self.relu(self.ligand1(ligand_cross))))

        receptor_cross_seq = ((out_receptor * receptor_mask_combined.unsqueeze(dim=2)).mean(dim=1) + receptor_seq_final) / 2
        receptor_cross_stru = ((out_receptor * receptor_mask_combined.unsqueeze(dim=2)).mean(dim=1) + receptor_graph_final) / 2
        receptor_cross = (receptor_cross_seq + receptor_cross_stru) / 2
        receptor_cross = self.receptor2(self.dropout(self.relu(self.receptor1(receptor_cross))))

        out = torch.cat((ligand_cross, receptor_cross), 1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        return out


# -------------------------------
# Load Ensemble Models
# -------------------------------
def load_models():
    model_paths = [
        "checkpoints/model_cv_(t300(5_fold))2_1_1.pth",
        "checkpoints/model_cv_(t300(5_fold))2_2_1.pth",
        "checkpoints/model_cv_(t300(5_fold))2_3_1.pth",
        "checkpoints/model_cv_(t300(5_fold))2_4_1.pth",
        "checkpoints/model_cv_(t300(5_fold))2_5_1.pth"
    ]

    models = []
    for path in model_paths:
        model = PPI().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models


# -------------------------------
# Inference Function
# -------------------------------
def run_inference(ligand_seq: str, receptor_seq: str, models):
    ligand = ProteinInference(sequence=ligand_seq)
    receptor = ProteinInference(sequence=receptor_seq)
    ligand_processed = ligand.process().to(device)
    receptor_processed = receptor.process().to(device)

    outputs = [m(ligand_processed, receptor_processed).item() for m in models]
    avg_output = sum(outputs) / len(outputs)
    return avg_output
