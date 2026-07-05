"""
Vector Database Module for Pistos.ai
Handles ChromaDB operations with local embeddings (no API calls).
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class LocalEmbeddings:
    """
    Simple local embeddings using sentence-transformers directly.
    Avoids HuggingFaceEmbeddings wrapper issues.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Use local cached model to avoid network calls
        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        local_model_path = cache_base / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        
        if local_model_path.exists():
            logger.info(f"Using cached model from: {local_model_path}")
            model_to_load = str(local_model_path)
        else:
            logger.info(f"Using model name: {model_name}")
            model_to_load = model_name
        
        logger.info(f"Loading embedding model: {model_to_load}")
        self.model = SentenceTransformer(model_to_load)
        logger.info("Embedding model loaded")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


class VectorDatabase:
    """
    Manages the ChromaDB vector database with local embeddings.
    All embeddings are generated locally for privacy and zero API costs.
    """
    
    def __init__(self, 
                 persist_directory: str = "data/vectorstore",
                 collection_name: str = "theological_documents",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.3):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace embedding model (local)
            similarity_threshold: Minimum similarity score for results
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing VectorDatabase at {self.persist_directory}")
        
        # Initialize ChromaDB with persistent storage
        # Use a more stable configuration with error recovery
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            logger.info("Using persistent ChromaDB client")
        except Exception as e:
            logger.warning(f"PersistentClient failed: {e}. Falling back to in-memory client...")
            # Fallback to in-memory client to avoid persistence issues
            try:
                self.client = chromadb.EphemeralClient(
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.warning("Using in-memory ChromaDB (data will not persist across restarts)")
            except Exception as ephemeral_error:
                logger.error(f"Both persistent and ephemeral clients failed: {ephemeral_error}")
                raise
        
        # Initialize local embeddings (no API calls)
        self.embeddings = LocalEmbeddings(model_name=embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Collection '{collection_name}' ready with {self.collection.count()} documents")
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector database...")
        
        # Prepare data for ChromaDB
        ids = []
        embeddings_list = []
        documents_list = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # Generate unique ID based on source and chunk index
            source = doc.metadata.get('source', 'unknown')
            chunk_idx = doc.metadata.get('chunk_index', i)
            doc_id = f"{Path(source).stem}_{chunk_idx}"
            
            # Generate embedding locally
            embedding = self.embeddings.embed_query(doc.page_content)
            
            ids.append(doc_id)
            embeddings_list.append(embedding)
            documents_list.append(doc.page_content)
            
            # Clean metadata for ChromaDB (only support certain types)
            clean_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            metadatas.append(clean_metadata)
            
            # Process in batches
            if len(ids) >= batch_size:
                self._add_batch(ids, embeddings_list, documents_list, metadatas)
                ids = []
                embeddings_list = []
                documents_list = []
                metadatas = []
        
        # Add remaining documents
        if ids:
            self._add_batch(ids, embeddings_list, documents_list, metadatas)
        
        logger.info(f"Successfully added {len(documents)} documents. Total in DB: {self.collection.count()}")
    
    def _add_batch(self, ids: List[str], embeddings: List[List[float]], 
                   documents: List[str], metadatas: List[Dict]) -> None:
        """
        Add a batch of documents to ChromaDB.
        
        Args:
            ids: Document IDs
            embeddings: List of embedding vectors
            documents: List of document texts
            metadatas: List of metadata dictionaries
        """
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def similarity_search(self, query: str, top_k: int = 5,
                         source_type: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents based on query.

        Args:
            query: The search query
            top_k: Number of results to return
            source_type: Filter by source type ('bible' or 'notes')

        Returns:
            List of tuples (Document, similarity_score)
        """
        logger.debug(f"Searching for: '{query[:50]}...' (top_k={top_k})")

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Build where filter
        where_filter = None
        if source_type:
            where_filter = {"source_type": source_type}

        # Query ChromaDB - get more results to filter by threshold
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3,  # Get 3x more results to filter by threshold
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to Document objects with scores
        retrieved_docs = []

        if results['documents'] and results['documents'][0]:
            for i, doc_content in enumerate(results['documents'][0]):
                # Convert distance to similarity score (cosine distance to similarity)
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity_score = 1 - distance  # Convert to similarity

                # Filter by threshold (more lenient)
                if similarity_score < self.similarity_threshold:
                    continue

                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )

                retrieved_docs.append((doc, similarity_score))

        # Sort by similarity score (highest first)
        retrieved_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        final_results = retrieved_docs[:top_k]

        logger.info(f"Retrieved {len(final_results)} documents with similarity >= {self.similarity_threshold}")

        return final_results
    
    def search_with_priority(self, query: str, top_k: int = 5,
                            source_priority: Dict[str, int] = None) -> List[Tuple[Document, float]]:
        """
        Search with source priority (e.g., Bible first, then Notes).

        Args:
            query: The search query
            top_k: Total number of results to return
            source_priority: Dictionary mapping source_type to priority (lower = higher priority)

        Returns:
            List of tuples (Document, similarity_score)
        """
        if source_priority is None:
            source_priority = {'bible': 1, 'notes': 2}

        logger.info(f"Searching with priority: {source_priority}")

        all_results = []

        # Search Bible first (priority 1)
        bible_results = self.similarity_search(
            query=query,
            top_k=top_k,
            source_type='bible'
        )
        
        # Add priority to metadata
        for doc, score in bible_results:
            doc.metadata['priority'] = 1
        all_results.extend(bible_results)

        # If we don't have enough Bible results, search notes
        if len(all_results) < top_k:
            notes_needed = top_k - len(all_results)
            notes_results = self.similarity_search(
                query=query,
                top_k=notes_needed * 2,  # Get extra to filter
                source_type='notes'
            )
            
            # Add priority to metadata
            for doc, score in notes_results:
                doc.metadata['priority'] = 2
            all_results.extend(notes_results)

        # Sort by priority first, then by similarity score
        all_results.sort(key=lambda x: (x[0].metadata.get('priority', 999), -x[1]))

        # Return top_k results
        final_results = all_results[:top_k]

        logger.info(f"Priority search returned {len(final_results)} documents")
        return final_results
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents from the vector database.
        
        Returns:
            List of all Document objects
        """
        count = self.collection.count()
        logger.info(f"Retrieving all {count} documents from database")
        
        if count == 0:
            return []
        
        # Get all documents in batches
        all_docs = []
        batch_size = 1000
        
        for i in range(0, count, batch_size):
            results = self.collection.get(
                limit=batch_size,
                offset=i,
                include=["documents", "metadatas"]
            )
            
            for j, doc_content in enumerate(results['documents']):
                metadata = results['metadatas'][j] if results['metadatas'] else {}
                doc = Document(page_content=doc_content, metadata=metadata)
                all_docs.append(doc)
        
        return all_docs
    
    def delete_collection(self) -> None:
        """
        Delete the current collection (use with caution).
        """
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection deleted and recreated")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        count = self.collection.count()
        
        # Get unique sources
        try:
            all_data = self.collection.get(include=["metadatas"])
            sources = set()
            source_types = {'bible': 0, 'notes': 0}
            
            for metadata in all_data['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])
                if 'source_type' in metadata:
                    source_types[metadata['source_type']] = source_types.get(metadata['source_type'], 0) + 1
            
            return {
                'total_documents': count,
                'unique_sources': len(sources),
                'source_breakdown': source_types,
                'collection_name': self.collection_name,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': str(self.persist_directory),
                'error': str(e)
            }
    
    def reset(self) -> None:
        """
        Reset the vector database (delete all documents).
        """
        logger.warning("Resetting vector database...")
        self.delete_collection()
        logger.info("Vector database reset complete")
