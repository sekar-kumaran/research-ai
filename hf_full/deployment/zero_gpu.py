import os
import numpy as np
import logging
from research_ai.retrieval.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

import spaces
ZEROGPU_AVAILABLE = True

# Global model instance for ZeroGPU
# ZeroGPU expects models to be placed on CUDA at module scope or via globals
global_embedding_model = None

def init_global_model(model_name: str, device: str = "cpu"):
    global global_embedding_model
    if global_embedding_model is None:
        logger.info("Initializing global ZeroGPU embedding model...")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        from sentence_transformers import SentenceTransformer
        local_only = os.getenv("MODEL_LOCAL_FILES_ONLY", "false").lower() == "true"
        hf_token = os.getenv("HF_TOKEN")
        global_embedding_model = SentenceTransformer(
            model_name, 
            local_files_only=local_only,
            token=hf_token
        )
        
        target = "cuda" if (device == "cuda" or (device == "auto" and ZEROGPU_AVAILABLE)) else "cpu"
        if target == "cuda":
            try:
                global_embedding_model.to("cuda")
                logger.info("Moved embedding model to CUDA.")
            except Exception as e:
                logger.warning(f"Failed to move model to CUDA: {e}")

@spaces.GPU
def _gpu_encode(texts: list[str], batch_size: int) -> np.ndarray:
    if global_embedding_model is None:
        raise RuntimeError("Global embedding model is not initialized.")
    return global_embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

class ZeroGPUEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "auto"):
        super().__init__(model_name)
        self.device_config = device
        self.use_gpu = (device == "cuda") or (device == "auto" and ZEROGPU_AVAILABLE)
        
        # Eagerly initialize global model
        init_global_model(model_name, device)

    @property
    def model(self):
        return global_embedding_model

    def _encode_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        if self.use_gpu:
            vectors = _gpu_encode(texts, batch_size)
        else:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)
