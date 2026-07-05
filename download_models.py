#!/usr/bin/env python3
"""
Download required models for offline use.
Run this once when you have internet connection.
"""

from huggingface_hub import snapshot_download
import os

print("Downloading embedding model for vector database...")
print("This may take a few minutes depending on your connection.")

try:
    # Download the embedding model
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir="data/models/embedding/all-MiniLM-L6-v2",
        local_dir_use_symlinks=False
    )
    print("✓ Embedding model downloaded successfully!")
    
    # Update config to use local model
    print("\nTo use the local model, update config.yaml:")
    print('embedding_model: "data/models/embedding/all-MiniLM-L6-v2"')
    
except Exception as e:
    print(f"Error downloading model: {e}")
    print("\nMake sure you have internet connection and try again.")
