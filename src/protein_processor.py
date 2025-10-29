import os
import re
import torch
from torch_geometric.data import Data
import numpy as np
import logging
import esm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESMEmbedder:

    def __init__(self, model_path: str, model_name: str, device: str = "cpu"):
        self.model_path = model_path
        self.model_name = model_name
        self.device = device

        loaded = False
        # Try local checkpoint first
        if os.path.isfile(self.model_path):
            logger.info(f"Loading ESM from local checkpoint: path={self.model_path}")
            try:
                model_data = torch.load(self.model_path, map_location="cpu")
                # load_model_and_alphabet_core expects the checkpoint dict
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_core(self.model_name, model_data)
                loaded = True
            except Exception as e:
                logger.warning(f"Failed to load local checkpoint '{self.model_path}': {e}")

        if not loaded:
            logger.info(f"Local checkpoint not found/failed. Attempting to load '{self.model_name}' from esm hub (this will download if needed).")
            try:
                # Prefer model-specific helper if present (e.g. esm.pretrained.esm2_t6_8M_UR50D)
                if hasattr(esm.pretrained, self.model_name):
                    self.model, self.alphabet = getattr(esm.pretrained, self.model_name)()
                else:
                    # Fallback to hub loader which accepts a model name
                    # (some esm versions expose load_model_and_alphabet_hub)
                    try:
                        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(self.model_name)
                    except Exception:
                        # Last fallback: try the generic esm.pretrained API if available
                        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                loaded = True
            except Exception as e:
                logger.error(f"Failed to load ESM model '{self.model_name}' from hub: {e}")
                raise

        # setup converter, eval and device
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(self.device)
        self.repr_layer = self._infer_repr_layer(self.model_name)

    @staticmethod
    def _infer_repr_layer(model_name: str) -> int:
        m = re.search(r"_t(\d+)_", model_name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        return 12

    def embed_sequence(self, seq: str) -> np.ndarray:
        _, _, batch_tokens = self.batch_converter([("protein", seq)])
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        rep = results["representations"][self.repr_layer][0, 1:-1].cpu().numpy()
        return rep

    def contact_map(self, seq: str) -> np.ndarray:
        """Try to get contact map from ESM; if unavailable, use simple chain adjacency."""
        _, _, batch_tokens = self.batch_converter([("protein", seq)])
        batch_tokens = batch_tokens.to(self.device)
        try:
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=True)
            contacts = results.get("contacts", None)
            if contacts is not None:
                return contacts[0].detach().cpu().numpy()
        except Exception as e:
            logger.warning(f"ESM contacts not available for model '{self.model_name}': {e}")

        # Fallback: simple chain adjacency
        L = len(seq)
        cm = np.zeros((L, L), dtype=np.float32)
        if L > 1:
            idx = np.arange(L - 1)
            cm[idx, idx + 1] = 1.0
            cm[idx + 1, idx] = 1.0
        return cm


_EMBEDDER_SINGLETON = None


def get_global_embedder() -> ESMEmbedder:
    """Return a singleton ESMEmbedder loaded from a local checkpoint.

    Env vars:
      - ESM_MODEL_PATH: path to .pt (default 'src/data/input/esm2_t6_8M_UR50D.pt')
      - ESM_MODEL_NAME: esm model key (default 'esm2_t6_8M_UR50D')
      - ESM_DEVICE: 'cuda' or 'cpu' (default auto)
    """
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is None:
        default_path = os.getenv("ESM_MODEL_PATH", os.path.join("src", "data", "input", "esm2_t6_8M_UR50D.pt"))
        model_name = os.getenv("ESM_MODEL_NAME", "esm2_t6_8M_UR50D")
        dev_env = os.getenv("ESM_DEVICE", "auto").lower()
        if dev_env == "cuda" or (dev_env == "auto" and torch.cuda.is_available()):
            device = "cuda"
        else:
            device = "cpu"
        _EMBEDDER_SINGLETON = ESMEmbedder(default_path, model_name, device)
    return _EMBEDDER_SINGLETON

class ProteinInference:
    def __init__(self, sequence):
        self.sequence = sequence
        # Use local-checkpoint embedder (singleton)
        self.embedder = get_global_embedder()
        self.esm_model = self.embedder.model
        self.alphabet = self.embedder.alphabet
        self.batch_converter = self.embedder.batch_converter

    def generate_esm_contact_map(self, sequence):
        """Generate contact map using ESM model with 0.5 threshold."""
        try:
            contact_map = self.embedder.contact_map(sequence)
            
            # Binarize contact map with 0.5 threshold
            binarized_map = (contact_map > 0.5).astype(np.int32)
            return binarized_map
        except Exception as e:
            logger.error(f"Error generating ESM contact map: {str(e)}")
            raise

    def generate_esm_embedding(self, sequence):
        """Generate protein embedding using ESM model."""
        try:
            return self.embedder.embed_sequence(sequence)
        except Exception as e:
            logger.error(f"Error generating ESM embedding: {str(e)}")
            raise

    def process(self):
        """Process single sequence for inference."""
        try:
            sequence = self.sequence[:998] if len(self.sequence) > 999 else self.sequence
            
            # Generate contact map using ESM
            contact_map = self.generate_esm_contact_map(sequence)
            sequence_length = len(sequence)
            
            if contact_map.shape != (sequence_length, sequence_length):
                logger.error(f"Contact map shape {contact_map.shape} does not match sequence length {sequence_length}")
                raise ValueError("Contact map shape mismatch")
            
            edges = np.argwhere(contact_map == 1)
            if edges.size > 0 and (edges.max() >= sequence_length):
                logger.error(f"Contact map contains invalid indices (max {edges.max()}) for sequence length {sequence_length}")
                raise ValueError("Invalid contact map indices")
            
            # Convert sequence to one-hot encoding
            try:
                one_hot_sequence = [char_to_one_hot(char) for char in sequence]
            except KeyError as e:
                logger.error(f"Invalid character in sequence: {e}")
                raise
            
            # Generate ESM embedding
            protein_emb = torch.tensor(self.generate_esm_embedding(sequence), dtype=torch.float32)
            
            # Create data object
            x = torch.tensor(one_hot_sequence, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            data = Data(
                x=x,
                edge_index=edge_index,
                emb=protein_emb,
                protein_len=x.size()[0]
            )
            
            logger.info(f"Successfully processed sequence for inference")
            return data
            
        except Exception as e:
            logger.error(f"Error in inference processing: {str(e)}")
            raise

# Amino acid to one-hot
def char_to_one_hot(char):
    mapping = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20  # Unknown or non-standard amino acid
    }
    return [mapping.get(char, 20)]  # Default to 'X' for unrecognized characters

# Example usage:
# inference = ProteinInference(sequence='ACDEFG')
# data = inference.process()