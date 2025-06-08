import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any
from huggingface_hub import HfApi, Repository, login
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor
from dotenv import load_dotenv
import gc
from tqdm import tqdm
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OscarEmbeddingProcessor:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 output_repo: str = None,
                 max_length: int = 8192,
                 embedding_dim: int = 1024,
                 batch_size: int = 16,
                 device: str = None):
        """
        Initialize the embedding processor
        
        Args:
            model_name: Hugging Face model name for embedding
            output_repo: Output repository name (e.g., "myduy/oscar-vi-embeddings")
            max_length: Maximum sequence length for tokenization
            embedding_dim: Embedding dimension
            batch_size: Batch size for processing
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.output_repo = output_repo
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize HF token
        self.hf_token = os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        # Login to Hugging Face
        login(token=self.hf_token)
        self.api = HfApi()
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side='left'
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Extract embeddings using last token pooling"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                batch_embeddings = self.last_token_pool(
                    outputs.last_hidden_state, 
                    batch_dict['attention_mask']
                )
                
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                # Convert to list and append
                embeddings.extend(batch_embeddings.cpu().float().tolist())
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return embeddings
    
    def process_chunk(self, chunk_name: str) -> List[Dict[str, Any]]:
        """
        Process a single chunk file
        
        Args:
            chunk_name: Name of the chunk file (e.g., "vi_meta_part_1_processed.json")
            
        Returns:
            List of dictionaries with id and embedding
        """
        logger.info(f"Processing chunk: {chunk_name}")
        
        try:
            # Load the chunk
            dataset = load_dataset(
                "myduy/oscar-vi", 
                data_files=f"processed_chunks/{chunk_name}",
                split="train"
            )
            
            # Extract content and prepare for embedding
            texts = [doc['content'] for doc in dataset]
            doc_ids = [doc['id'] for doc in dataset]
            
            logger.info(f"Found {len(texts)} documents in {chunk_name}")
            
            # Generate embeddings
            embeddings = self.embed_texts(texts)
            
            # Prepare output format
            embedded_docs = []
            for doc_id, embedding in zip(doc_ids, embeddings):
                embedded_docs.append({
                    "id": f"{chunk_name}_{doc_id}",
                    "embedding": embedding
                })
            
            logger.info(f"Generated {len(embedded_docs)} embeddings for {chunk_name}")
            return embedded_docs
            
        except Exception as e:
            logger.error(f"Error processing {chunk_name}: {str(e)}")
            return []
    
    def save_embeddings_chunk(self, embeddings: List[Dict[str, Any]], chunk_name: str):
        """
        Save embeddings for a chunk to HuggingFace
        
        Args:
            embeddings: List of embedding dictionaries
            chunk_name: Original chunk name
        """
        if not embeddings:
            logger.warning(f"No embeddings to save for {chunk_name}")
            return
        
        try:
            # Create output filename
            output_filename = chunk_name.replace('.json', '_embeddings.json')
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=2)
                temp_file = f.name
            
            # Upload to HuggingFace
            self.api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo=f"embeddings/{output_filename}",
                repo_id=self.output_repo,
                token=self.hf_token,
                repo_type="dataset"
            )
            
            # Clean up temp file
            os.unlink(temp_file)
            
            logger.info(f"Uploaded embeddings for {chunk_name} to {self.output_repo}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings for {chunk_name}: {str(e)}")
    
    def get_chunk_list(self) -> List[str]:
        """Get list of chunk files from the oscar-vi dataset"""
        try:
            # Get file list from the repository
            repo_files = self.api.list_repo_files(
                repo_id="myduy/oscar-vi",
                repo_type="dataset"
            )
            
            # Filter for processed_chunks JSON files
            chunk_files = [
                f.split('/')[-1] for f in repo_files 
                if f.startswith('processed_chunks/') and f.endswith('.json')
            ]
            
            logger.info(f"Found {len(chunk_files)} chunk files")
            return sorted(chunk_files)
            
        except Exception as e:
            logger.error(f"Error getting chunk list: {str(e)}")
            return []
    
    def create_output_repo(self):
        """Create output repository if it doesn't exist"""
        try:
            self.api.create_repo(
                repo_id=self.output_repo,
                repo_type="dataset",
                exist_ok=True,
                token=self.hf_token
            )
            logger.info(f"Repository {self.output_repo} is ready")
            
            # Create README
            readme_content = f"""# Oscar-VI Embeddings

This dataset contains embeddings for the Vietnamese OSCAR corpus using {self.model_name}.

## Structure

Each file in the `embeddings/` directory corresponds to a chunk from the original oscar-vi dataset.

Format: 
```json
[
  {{
    "id": "chunk_name_doc_id",
    "embedding": [0.1, 0.2, ...]
  }}
]
```

- Embedding dimension: {self.embedding_dim}
- Model used: {self.model_name}
- Max sequence length: {self.max_length}
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(readme_content)
                temp_readme = f.name
            
            self.api.upload_file(
                path_or_fileobj=temp_readme,
                path_in_repo="README.md",
                repo_id=self.output_repo,
                token=self.hf_token,
                repo_type="dataset"
            )
            
            os.unlink(temp_readme)
            
        except Exception as e:
            logger.error(f"Error creating output repository: {str(e)}")
    
    def process_all_chunks(self):
        """Process all chunks in the oscar-vi dataset"""
        # Create output repository
        if self.output_repo:
            self.create_output_repo()
        
        # Get list of chunks
        chunk_files = self.get_chunk_list()
        
        if not chunk_files:
            logger.error("No chunk files found")
            return
        
        logger.info(f"Processing {len(chunk_files)} chunks")
        
        for i, chunk_name in enumerate(chunk_files, 1):
            logger.info(f"Processing {i}/{len(chunk_files)}: {chunk_name}")
            
            try:
                # Process chunk
                embeddings = self.process_chunk(chunk_name)
                
                # Save embeddings
                if self.output_repo and embeddings:
                    self.save_embeddings_chunk(embeddings, chunk_name)
                
                # Force garbage collection
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info(f"Completed {chunk_name}")
                
            except Exception as e:
                logger.error(f"Failed to process {chunk_name}: {str(e)}")
                continue
        
        logger.info("All chunks processed!")

def main():
    """Main function to run the embedding processor"""
    # Configuration
    output_repo = "myduy/oscar-vi-embeddings"  # Change this to your desired repo name
    
    # Initialize processor
    processor = OscarEmbeddingProcessor(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        output_repo=output_repo,
        max_length=8192,
        embedding_dim=1024,
        batch_size=8,  # Adjust based on your GPU memory
    )
    
    # Process all chunks
    processor.process_all_chunks()

if __name__ == "__main__":
    main() 