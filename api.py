from datetime import datetime
from flask import Flask, Response, request, jsonify
import torch
import torch.nn as nn
from src.protein_processor import ProteinInference
from src.model import Protein_feature_extraction, cross_attention

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

hidden_dim = 128

# -----------------------------
# Define PPI Model
# -----------------------------
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

        context_layer, attention_score = self.cross_attention(
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


# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load Pretrained Models
# -----------------------------
def _load_checkpoint_with_key_remap(model: nn.Module, ckpt_path: str, device: torch.device):
    """Load a checkpoint while remapping legacy key names to current model.

    - Renames 'rna' -> 'ligand' and 'mole' -> 'receptor' in cross-attention blocks
    - Strips an optional leading 'module.' (from DataParallel)
    - Loads with strict=False so non-matching keys don't crash the app
    """
    state = torch.load(ckpt_path, map_location=device)
    # Some checkpoints wrap the state_dict
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']

    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {ckpt_path}")

    remapped = {}
    for k, v in state.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        # Remap legacy naming used in provided checkpoints
        new_k = new_k.replace('rna', 'ligand')
        new_k = new_k.replace('mole', 'receptor')
        remapped[new_k] = v

    incompatible = model.load_state_dict(remapped, strict=False)
    # Log what didn't match to help debugging but don't crash
    try:
        missing = getattr(incompatible, 'missing_keys', [])
        unexpected = getattr(incompatible, 'unexpected_keys', [])
        if missing or unexpected:
            print(f"Checkpoint load for {ckpt_path}: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                print(f"Missing keys (first 10): {missing[:10]}")
            if unexpected:
                print(f"Unexpected keys (first 10): {unexpected[:10]}")
    except Exception:
        # Older torch may return None when strict=False; ignore
        pass
    model.eval()
    return model

try:
    model1 = PPI().to(device)
    model2 = PPI().to(device)
    model3 = PPI().to(device)
    model4 = PPI().to(device)
    model5 = PPI().to(device)

    model1 = _load_checkpoint_with_key_remap(model1, "save/model_cv_(t300(5_fold))2_1_1.pth", device)
    model2 = _load_checkpoint_with_key_remap(model2, "save/model_cv_(t300(5_fold))2_2_1.pth", device)
    model3 = _load_checkpoint_with_key_remap(model3, "save/model_cv_(t300(5_fold))2_3_1.pth", device)
    model4 = _load_checkpoint_with_key_remap(model4, "save/model_cv_(t300(5_fold))2_4_1.pth", device)
    model5 = _load_checkpoint_with_key_remap(model5, "save/model_cv_(t300(5_fold))2_5_1.pth", device)
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise


# -----------------------------
# Inference Endpoint
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input Validation
        if not request.json or 'ligand' not in request.json or 'receptor' not in request.json:
            return jsonify({'error': 'Missing "ligand" or "receptor" in request'}), 400

        ligand_seq = request.json['ligand']
        receptor_seq = request.json['receptor']

        if not isinstance(ligand_seq, str) or not ligand_seq:
            return jsonify({'error': 'Ligand must be a non-empty string'}), 400
        if not isinstance(receptor_seq, str) or not receptor_seq:
            return jsonify({'error': 'Receptor must be a non-empty string'}), 400

        # Process input sequences
        try:
            ligand_proc = ProteinInference(sequence=ligand_seq).process()
            receptor_proc = ProteinInference(sequence=receptor_seq).process()
        except Exception as e:
            return jsonify({'error': f'Error processing sequences: {str(e)}'}), 400

        # Model inference (ensemble average)
        with torch.no_grad():
            preds = [
                model1(ligand_proc.to(device), receptor_proc.to(device)).item(),
                model2(ligand_proc.to(device), receptor_proc.to(device)).item(),
                model3(ligand_proc.to(device), receptor_proc.to(device)).item(),
                model4(ligand_proc.to(device), receptor_proc.to(device)).item(),
                model5(ligand_proc.to(device), receptor_proc.to(device)).item(),
            ]
            final_pred = sum(preds) / len(preds)

        return jsonify({'prediction': final_pred}), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.route('/health/<sample>', methods=['POST'])
def health(sample) -> Response:
    if sample == "hi":
        date = datetime.now().strftime("%H:%M:%S")
        return f"Hello {date}"
    else:
        return jsonify({'error': "Unauthorized access"})


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5060)
