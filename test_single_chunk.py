import os
import json
import torch
import logging
from dotenv import load_dotenv
from embedding_processor import OscarEmbeddingProcessor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_chunk():
    """Test processing a single chunk"""
    
    # Configuration
    output_repo = "myduy/oscar-vi-embeddings-test"  # Test repo
    
    # Initialize processor
    processor = OscarEmbeddingProcessor(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        output_repo=output_repo,
        max_length=8192,
        embedding_dim=1024,
        batch_size=4,  # Smaller batch for testing
    )
    
    # Create output repository
    processor.create_output_repo()
    
    # Test with the first chunk we can find
    chunk_files = processor.get_chunk_list()
    
    if not chunk_files:
        logger.error("No chunk files found")
        return
    
    test_chunk = chunk_files[0]  # Take the first chunk
    logger.info(f"Testing with chunk: {test_chunk}")
    
    # Process the test chunk
    embeddings = processor.process_chunk(test_chunk)
    
    if embeddings:
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        logger.info(f"First embedding shape: {len(embeddings[0]['embedding'])}")
        logger.info(f"Sample ID: {embeddings[0]['id']}")
        
        # Save to local file first for inspection
        output_file = f"test_{test_chunk.replace('.json', '_embeddings.json')}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings[:5], f, ensure_ascii=False, indent=2)  # Save only first 5 for inspection
        
        logger.info(f"Sample embeddings saved to {output_file}")
        
        # Save to HuggingFace
        processor.save_embeddings_chunk(embeddings, test_chunk)
        
    else:
        logger.error("Failed to generate embeddings")

if __name__ == "__main__":
    test_single_chunk() 