#!/usr/bin/env python3
"""
Document Update CLI for Pistos.ai
Command-line tool to process and update documents in the vector database.

Usage:
    python update_database.py              # Process all files in uploads folder
    python update_database.py --reset      # Reset database and reprocess all
    python update_database.py --file path  # Process a single file
    python update_database.py --stats      # Show database statistics
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import yaml
from tqdm import tqdm

from document_processor import DocumentProcessor
from vector_database import VectorDatabase


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def process_documents(config: dict, uploads_dir: str, reset: bool = False) -> int:
    """
    Process documents from uploads directory.
    
    Args:
        config: Configuration dictionary
        uploads_dir: Path to uploads directory
        reset: Whether to reset database before processing
        
    Returns:
        Number of chunks added
    """
    logger = logging.getLogger(__name__)
    
    doc_config = config.get('document_processing', {})
    vector_config = config.get('vectorstore', {})
    
    # Initialize components
    logger.info("Initializing document processor...")
    processor = DocumentProcessor(
        chunk_size=doc_config.get('chunk_size', 750),
        chunk_overlap_percent=doc_config.get('chunk_overlap', 0.15)
    )
    
    logger.info("Initializing vector database...")
    vector_db = VectorDatabase(
        persist_directory=vector_config.get('persist_directory', 'data/vectorstore'),
        collection_name=vector_config.get('collection_name', 'theological_documents'),
        embedding_model=vector_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        similarity_threshold=vector_config.get('similarity_threshold', 0.3)
    )
    
    # Reset if requested
    if reset:
        logger.warning("Resetting vector database...")
        vector_db.reset()
    
    # Get initial count
    initial_count = vector_db.collection.count()
    logger.info(f"Initial document count: {initial_count}")
    
    # Process directory
    uploads_path = Path(uploads_dir)
    
    if not uploads_path.exists():
        logger.error(f"Uploads directory not found: {uploads_path}")
        return 0
    
    supported_extensions = doc_config.get('supported_extensions', ['.pdf', '.txt', '.md'])
    
    # Get list of files to process
    files_to_process = [
        f for f in uploads_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
        and not f.name.startswith('~') and not f.name.startswith('.')
    ]
    
    if not files_to_process:
        logger.warning(f"No supported files found in {uploads_path}")
        return 0
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process each file with progress bar
    total_chunks = 0
    for file_path in tqdm(files_to_process, desc="Processing documents"):
        try:
            logger.debug(f"Processing: {file_path.name}")
            
            # Load document
            text = processor.load_document(str(file_path))
            
            # Determine source type
            source_type = processor.determine_source_type(str(file_path))
            
            # Create metadata
            metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'source_type': source_type
            }
            
            # Chunk the text
            chunks = processor.chunk_text(text, source_type, metadata)
            
            # Add to vector database
            if chunks:
                vector_db.add_documents(chunks)
                total_chunks += len(chunks)
                logger.debug(f"Added {len(chunks)} chunks from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue
    
    # Get final count
    final_count = vector_db.collection.count()
    
    logger.info("=" * 50)
    logger.info("Document Processing Complete")
    logger.info("=" * 50)
    logger.info(f"Files processed: {len(files_to_process)}")
    logger.info(f"Chunks added: {total_chunks}")
    logger.info(f"Total documents in database: {final_count}")
    logger.info("=" * 50)
    
    return total_chunks


def process_single_file(config: dict, file_path: str) -> int:
    """
    Process a single file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to the file
        
    Returns:
        Number of chunks added
    """
    logger = logging.getLogger(__name__)
    
    doc_config = config.get('document_processing', {})
    vector_config = config.get('vectorstore', {})
    
    # Initialize components
    processor = DocumentProcessor(
        chunk_size=doc_config.get('chunk_size', 750),
        chunk_overlap_percent=doc_config.get('chunk_overlap', 0.15)
    )
    
    vector_db = VectorDatabase(
        persist_directory=vector_config.get('persist_directory', 'data/vectorstore'),
        collection_name=vector_config.get('collection_name', 'theological_documents'),
        embedding_model=vector_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        similarity_threshold=vector_config.get('similarity_threshold', 0.3)
    )
    
    # Process file
    try:
        logger.info(f"Processing file: {file_path}")
        
        text = processor.load_document(file_path)
        source_type = processor.determine_source_type(file_path)
        
        metadata = {
            'source': file_path,
            'filename': Path(file_path).name,
            'source_type': source_type
        }
        
        chunks = processor.chunk_text(text, source_type, metadata)
        
        if chunks:
            initial_count = vector_db.collection.count()
            vector_db.add_documents(chunks)
            final_count = vector_db.collection.count()
            
            logger.info(f"Added {final_count - initial_count} chunks from {Path(file_path).name}")
            logger.info(f"Total documents in database: {final_count}")
            
            return len(chunks)
        else:
            logger.warning("No chunks created")
            return 0
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return 0


def show_stats(config: dict) -> None:
    """
    Show database statistics.
    
    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    vector_config = config.get('vectorstore', {})
    
    vector_db = VectorDatabase(
        persist_directory=vector_config.get('persist_directory', 'data/vectorstore'),
        collection_name=vector_config.get('collection_name', 'theological_documents'),
        embedding_model=vector_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        similarity_threshold=vector_config.get('similarity_threshold', 0.3)
    )
    
    stats = vector_db.get_stats()
    
    print("\n" + "=" * 50)
    print("Vector Database Statistics")
    print("=" * 50)
    print(f"Collection Name: {stats.get('collection_name', 'N/A')}")
    print(f"Persist Directory: {stats.get('persist_directory', 'N/A')}")
    print(f"Total Documents: {stats.get('total_documents', 0)}")
    print(f"Unique Sources: {stats.get('unique_sources', 0)}")
    
    source_breakdown = stats.get('source_breakdown', {})
    if source_breakdown:
        print("\nSource Breakdown:")
        for source_type, count in source_breakdown.items():
            print(f"  - {source_type}: {count}")
    
    if 'error' in stats:
        print(f"\nError: {stats['error']}")
    
    print("=" * 50 + "\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Pistos.ai Document Update CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python update_database.py                  # Process all files in uploads folder
    python update_database.py --reset          # Reset database and reprocess all
    python update_database.py --file doc.pdf   # Process a single file
    python update_database.py --stats          # Show database statistics
    python update_database.py --dir /path      # Process files from specific directory
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--reset', '-r',
        action='store_true',
        help='Reset database before processing'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Process a single file'
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='data/uploads',
        help='Directory containing documents to process (default: data/uploads)'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show database statistics'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Show stats only
        if args.stats:
            show_stats(config)
            return
        
        # Process single file
        if args.file:
            chunks = process_single_file(config, args.file)
            if chunks > 0:
                logger.info(f"Successfully processed {args.file}: {chunks} chunks added")
            else:
                logger.warning(f"No chunks added from {args.file}")
            return
        
        # Process directory
        logger.info(f"Processing documents from: {args.dir}")
        chunks = process_documents(config, args.dir, reset=args.reset)
        
        if chunks > 0:
            logger.info(f"Successfully processed {chunks} chunks")
        else:
            logger.warning("No chunks were added")
            
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
