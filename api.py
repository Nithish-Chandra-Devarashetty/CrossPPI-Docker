from flask import Flask, request, jsonify, Response
import torch
import os
import json
import traceback
import shutil
from datetime import datetime
from src.protein_processor import ProteinInference
from src.model import Protein_feature_extraction, cross_attention
import torch.nn as nn

app = Flask(__name__)

UPLOAD_FOLDER = 'src/data/input'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
# Load Pretrained Ensemble Models
# -------------------------------
model_paths = [
    "srcsave/model_cv_(t300(5_fold))2_1_1.pth",
    "src/save/model_cv_(t300(5_fold))2_2_1.pth",
    "src/save/model_cv_(t300(5_fold))2_3_1.pth",
    "src/save/model_cv_(t300(5_fold))2_4_1.pth",
    "src/save/model_cv_(t300(5_fold))2_5_1.pth"
]

models = []
for path in model_paths:
    model = PPI().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    models.append(model)

# -------------------------------
# Inference Helper Function
# -------------------------------
def run_inference(ligand_seq: str, receptor_seq: str):
    ligand = ProteinInference(sequence=ligand_seq)
    receptor = ProteinInference(sequence=receptor_seq)
    ligand_processed = ligand.process().to(device)
    receptor_processed = receptor.process().to(device)

    outputs = [m(ligand_processed, receptor_processed).item() for m in models]
    avg_output = sum(outputs) / len(outputs)
    return avg_output

# -------------------------------
# /score Endpoint
# -------------------------------
@app.route('/score', methods=['POST'])
def score() -> Response:
    reqId = None
    try:
        data = request.form
        required_fields = ['reqId', 'ligand_seq', 'receptor_seq']
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")

        reqId = data.get('reqId')
        ligand_seq = data.get('ligand_seq')
        receptor_seq = data.get('receptor_seq')

        # Save input for traceability
        with open(f"{UPLOAD_FOLDER}/{reqId}_input.json", "w") as f:
            json.dump({'ligand_seq': ligand_seq, 'receptor_seq': receptor_seq}, f)

        # Run inference
        result = run_inference(ligand_seq, receptor_seq)

        return jsonify({
            "message": "Inference completed successfully",
            "reqId": reqId,
            "predicted_affinity": result
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "message": "Inference failed",
            "error": str(e)
        }), 500

    finally:
        if reqId:
            for file in os.listdir(UPLOAD_FOLDER):
                if file.startswith(reqId + "_"):
                    os.remove(os.path.join(UPLOAD_FOLDER, file))

# -------------------------------
# /health Endpoint
# -------------------------------
@app.route('/health/<sample>', methods=['GET'])
def health(sample) -> Response:
    if sample == "hi":
        date = datetime.now().strftime("%H:%M:%S")
        return f"Hello {date}"
    return jsonify({'error': "Unauthorized access"}), 403

# -------------------------------
# Run Server
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
