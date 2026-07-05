#!/usr/bin/env python3
"""
Fast document processor for large PDFs.
Uses pypdf for faster extraction (sacrifices some accuracy for speed).
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pypdf
import yaml
from tqdm import tqdm

from document_processor import DocumentProcessor
from vector_database import VectorDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fast_process_pdf(file_path: str, chunk_size: int = 1000) -> list:
    """
    Fast PDF processing using pypdf only.
    
    Args:
        file_path: Path to PDF
        chunk_size: Characters per chunk
        
    Returns:
        List of text chunks
    """
    file_path = Path(file_path)
    logger.info(f"Fast processing: {file_path.name}")
    
    chunks = []
    current_text = ""
    
    with open(file_path, 'rb') as f:
        pdf = pypdf.PdfReader(f)
        total_pages = len(pdf.pages)
        logger.info(f"Processing {total_pages} pages...")
        
        for i, page in enumerate(tqdm(pdf.pages, desc="Extracting")):
            text = page.extract_text()
            if text:
                current_text += text + "\n"
                
                # Create chunks when we have enough text
                if len(current_text) >= chunk_size * 2:
                    # Split into chunks
                    for j in range(0, len(current_text), chunk_size):
                        chunk = current_text[j:j + chunk_size]
                        if chunk.strip():
                            chunks.append({
                                'content': chunk,
                                'source': str(file_path),
                                'pages': f"{i+1}"
                            })
                    current_text = current_text[-chunk_size:]  # Keep overlap
        
        # Process remaining text
        if current_text.strip():
            for j in range(0, len(current_text), chunk_size):
                chunk = current_text[j:j + chunk_size]
                if chunk.strip():
                    chunks.append({
                        'content': chunk,
                        'source': str(file_path),
                        'pages': f"{total_pages}"
                    })
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast PDF processor")
    parser.add_argument("--file", "-f", required=True, help="PDF file to process")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    args = parser.parse_args()
    
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    vector_config = config.get('vectorstore', {})
    
    # Initialize vector DB
    logger.info("Initializing vector database...")
    vector_db = VectorDatabase(
        persist_directory=vector_config.get('persist_directory', 'data/vectorstore'),
        collection_name=vector_config.get('collection_name', 'theological_documents')
    )
    
    # Process PDF
    chunks = fast_process_pdf(args.file, args.chunk_size)
    
    # Add to vector DB
    if chunks:
        logger.info(f"Adding {len(chunks)} chunks to database...")
        
        from langchain_core.documents import Document
        
        docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'source': chunk['source'],
                    'pages': chunk['pages'],
                    'chunk_index': i,
                    'source_type': 'bible' if 'bible' in chunk['source'].lower() else 'notes'
                }
            )
            docs.append(doc)
        
        vector_db.add_documents(docs)
        logger.info(f"Done! Total documents in DB: {vector_db.collection.count()}")
