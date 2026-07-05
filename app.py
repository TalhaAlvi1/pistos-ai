#!/usr/bin/env python3
"""
Pistos.ai - Main Application Entry Point (Streamlit Version)
Private RAG Chatbot for Theological Documents

Usage:
    streamlit run app_streamlit.py              # Start the web interface
    streamlit run app_streamlit.py -- --init    # Initialize and process documents first
"""

import os
import sys
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rag_pipeline import RAGPipeline
from web_interface import WebInterface


def setup_logging(config: dict) -> logging.Logger:
    """Set up logging based on configuration."""
    log_config = config.get('logging', {})
    
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_file = log_config.get('file', 'logs/pistos_chatbot.log')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def initialize_pipeline(config: dict, uploads_dir: str = None) -> RAGPipeline:
    """Initialize the RAG pipeline and optionally process documents."""
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(config)
    
    logger.info("Loading LLM model (this may take a minute)...")
    pipeline.initialize()
    
    logger.info("Pipeline initialized successfully")
    
    # Process documents if uploads directory is specified
    if uploads_dir:
        uploads_path = Path(uploads_dir)
        if uploads_path.exists():
            logger.info(f"Processing documents from: {uploads_path}")
            chunks = pipeline.process_and_index_documents(str(uploads_path))
            logger.info(f"Indexed {chunks} document chunks")
        else:
            logger.warning(f"Uploads directory not found: {uploads_path}")
    
    return pipeline


def main():
    """Main entry point for the application."""
    try:
        # Check for --init flag
        init_mode = '--init' in sys.argv
        
        # Load configuration
        print(f"Loading configuration from: config.yaml")
        config = load_config('config.yaml')
        
        # Set up logging
        logger = setup_logging(config)
        logger.info("Pistos.ai starting (Streamlit version)...")
        
        # Initialize pipeline
        uploads_dir = 'data/uploads' if init_mode else None
        pipeline = initialize_pipeline(config, uploads_dir)
        
        # Create and launch interface
        interface = WebInterface(pipeline, config)
        interface.launch()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logging.exception("Unexpected error")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
