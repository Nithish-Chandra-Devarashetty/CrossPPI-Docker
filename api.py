from flask import Flask, request, jsonify, Response
import os
import json
import traceback
from datetime import datetime

from src.inference import run_inference, load_models

app = Flask(__name__)

UPLOAD_FOLDER = 'src/data/input'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models once at startup
models = load_models()

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
        result = run_inference(ligand_seq, receptor_seq, models)

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
