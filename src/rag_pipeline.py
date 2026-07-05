"""
RAG Pipeline Module for Pistos.ai
Implements Retrieval-Augmented Generation with strict context adherence.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
from langchain_core.documents import Document

from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from llm_manager_gguf import LLMManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for theological document Q&A.
    Enforces strict context adherence - never answers from model's prior knowledge.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        
        # Extract configuration sections
        model_config = config.get('model', {})
        doc_config = config.get('document_processing', {})
        vector_config = config.get('vectorstore', {})
        response_config = config.get('response', {})
        rag_config = config.get('rag', {})
        
        # Initialize components
        logger.info("Initializing RAG Pipeline components...")
        
        # Document Processor
        self.doc_processor = DocumentProcessor(
            chunk_size=doc_config.get('chunk_size', 750),
            chunk_overlap_percent=doc_config.get('chunk_overlap', 0.15)
        )
        
        # Vector Database
        self.vector_db = VectorDatabase(
            persist_directory=vector_config.get('persist_directory', 'data/vectorstore'),
            collection_name=vector_config.get('collection_name', 'theological_documents'),
            embedding_model=vector_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            similarity_threshold=vector_config.get('similarity_threshold', 0.3)
        )
        
        # LLM Manager
        self.llm = LLMManager(
            model_path=model_config.get('model_path', 'data/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'),
            context_size=model_config.get('context_size', 4096),
            gpu_layers=model_config.get('gpu_layers', 0),
            threads=model_config.get('threads', 4)
        )
        
        # Response settings
        self.fallback_message = response_config.get(
            'fallback_message',
            "I cannot find the answer in the provided texts."
        )
        self.tone = response_config.get('tone', 'scholarly')
        self.length = response_config.get('length', 'detailed')
        self.formatting = response_config.get('formatting', {})
        
        # RAG settings
        self.strict_mode = rag_config.get('strict_context_adherence', True)
        self.max_context_tokens = rag_config.get('max_context_tokens', 3000)
        self.temperature = rag_config.get('temperature', 0.3)
        self.top_p = rag_config.get('top_p', 0.9)
        self.max_tokens = rag_config.get('max_tokens', 512)
        
        # Source priority
        self.source_priority = doc_config.get('source_priority', {'bible': 1, 'notes': 2})
        
        logger.info("RAG Pipeline initialized successfully")
    
    def initialize(self) -> None:
        """
        Initialize all components (load model, etc.)
        """
        logger.info("Initializing RAG Pipeline...")
        self.llm.load_model()
        logger.info(f"Vector DB has {self.vector_db.collection.count()} documents")
        logger.info("RAG Pipeline ready")
    
    def process_and_index_documents(self, directory_path: str) -> int:
        """
        Process documents from a directory and add to vector database.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            Number of chunks added to the database
        """
        logger.info(f"Processing documents from: {directory_path}")
        
        supported_extensions = self.config.get('document_processing', {}).get(
            'supported_extensions', ['.pdf', '.txt', '.md']
        )
        
        # Process documents
        documents = self.doc_processor.process_directory(
            directory_path=directory_path,
            supported_extensions=supported_extensions
        )
        
        if not documents:
            logger.warning("No documents were processed")
            return 0
        
        # Add to vector database
        initial_count = self.vector_db.collection.count()
        self.vector_db.add_documents(documents)
        final_count = self.vector_db.collection.count()
        
        added_count = final_count - initial_count
        logger.info(f"Added {added_count} chunks to vector database")
        
        return added_count
    
    def process_single_file(self, file_path: str) -> int:
        """
        Process a single file and add to vector database.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of chunks added
        """
        logger.info(f"Processing single file: {file_path}")
        
        documents = self.doc_processor.process_single_file(file_path)
        
        if not documents:
            logger.warning("No documents were processed")
            return 0
        
        initial_count = self.vector_db.collection.count()
        self.vector_db.add_documents(documents)
        final_count = self.vector_db.collection.count()
        
        added_count = final_count - initial_count
        logger.info(f"Added {added_count} chunks from {file_path}")
        
        return added_count
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt from configuration.
        
        Returns:
            Formatted system prompt string
        """
        base_prompt = self.config.get('system_prompt', '')
        
        # Format with response settings
        system_prompt = base_prompt.format(
            fallback_message=self.fallback_message,
            tone=self.tone,
            length=self.length
        )
        
        return system_prompt
    
    def _format_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            retrieved_docs: List of (Document, score) tuples
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            source_type = doc.metadata.get('source_type', 'unknown')
            source = doc.metadata.get('filename', 'unknown')
            
            context_part = (
                f"[Source {i}] (Type: {source_type}, File: {source}, Relevance: {score:.2f})\n"
                f"{doc.page_content}"
            )
            context_parts.append(context_part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _check_context_has_answer(self, question: str, context: str) -> bool:
        """
        Check if the retrieved context likely contains an answer to the question.
        Uses simple heuristics to determine if context is relevant.
        """
        if not context or not question:
            return False

        # If we have substantial context, let the LLM handle it
        if len(context) > 100:
            return True

        # Check for keyword overlap
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                      'those', 'am', 'and', 'but', 'if', 'or', 'because', 'until',
                      'while', 'about', 'against', 'how', 'any', 'all', 'each',
                      'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                      'say', 'says', 'said', 'does', 'do', 'did'}

        question_words = question_words - stop_words
        context_words = context_words - stop_words

        if not question_words:
            return True

        overlap = question_words & context_words
        overlap_ratio = len(overlap) / max(1, len(question_words))

        # More lenient threshold - let LLM decide
        return overlap_ratio >= 0.05
    
    def query(self, question: str, use_streaming: bool = False) -> str:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: User's question
            use_streaming: Whether to use streaming response
            
        Returns:
            AI response
        """
        logger.info(f"Processing question: '{question[:50]}...'")
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_db.search_with_priority(
            query=question,
            top_k=self.config.get('vectorstore', {}).get('top_k', 5),
            source_priority=self.source_priority
        )
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Format context
        context = self._format_context(retrieved_docs)
        
        # Check if context is sufficient (strict mode)
        if self.strict_mode and not self._check_context_has_answer(question, context):
            logger.info("Context does not contain sufficient information")
            return self.fallback_message
        
        # If no context at all, return fallback
        if not context:
            logger.info("No context retrieved")
            return self.fallback_message
        
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Format full prompt for Llama 3
        full_prompt = self.llm.format_prompt(
            system_prompt=system_prompt,
            user_question=question,
            context=context
        )
        
        # Generate response
        if use_streaming:
            # For streaming, we need to collect and return at the end
            # Streaming is handled separately in query_stream
            return self.llm.generate(
                prompt=full_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop_sequences=["<|eot_id|>", "<|end_of_text|>"]
            )
        else:
            response = self.llm.generate(
                prompt=full_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop_sequences=["<|eot_id|>", "<|end_of_text|>"]
            )
            
            logger.info(f"Generated response: {response[:100]}...")
            return response
    
    def query_stream(self, question: str) -> Generator[str, None, None]:
        """
        Query the RAG pipeline with streaming response.
        
        Args:
            question: User's question
            
        Yields:
            Chunks of the response
        """
        logger.info(f"Processing streaming question: '{question[:50]}...'")
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_db.search_with_priority(
            query=question,
            top_k=self.config.get('vectorstore', {}).get('top_k', 5),
            source_priority=self.source_priority
        )
        
        # Format context
        context = self._format_context(retrieved_docs)
        
        # Check if context is sufficient
        if self.strict_mode and not self._check_context_has_answer(question, context):
            yield self.fallback_message
            return
        
        if not context:
            yield self.fallback_message
            return
        
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Format full prompt
        full_prompt = self.llm.format_prompt(
            system_prompt=system_prompt,
            user_question=question,
            context=context
        )
        
        # Stream the response
        try:
            for chunk in self.llm.generate_stream(
                prompt=full_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop_sequences=["<|eot_id|>", "<|end_of_text|>"]
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"[Error generating response: {str(e)}]"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        db_stats = self.vector_db.get_stats()
        model_info = self.llm.get_model_info()
        
        return {
            'database': db_stats,
            'model': model_info,
            'config': {
                'strict_mode': self.strict_mode,
                'tone': self.tone,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        }
    
    def reset_database(self) -> None:
        """
        Reset the vector database (delete all documents).
        """
        logger.warning("Resetting vector database...")
        self.vector_db.reset()
        logger.info("Vector database reset complete")
    
    def close(self) -> None:
        """
        Clean up resources.
        """
        logger.info("Closing RAG Pipeline...")
        self.llm.unload_model()
        logger.info("RAG Pipeline closed")
