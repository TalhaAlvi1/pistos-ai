"""
LLM Integration Module for Pistos.ai
Handles Llama 3 inference using llama-cpp-python with quantized models.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Generator
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages Llama 3 inference using llama-cpp-python.
    Supports quantized GGUF models for efficient CPU/GPU inference.
    """
    
    def __init__(self,
                 model_path: str = "data/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
                 context_size: int = 4096,
                 gpu_layers: int = 0,
                 threads: int = 4):
        """
        Initialize the LLM manager.
        
        Args:
            model_path: Path to the GGUF model file
            context_size: Maximum context window size
            gpu_layers: Number of layers to offload to GPU (0 for CPU-only)
            threads: Number of CPU threads for inference
        """
        self.model_path = Path(model_path)
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.threads = threads
        
        self.model = None
        
        logger.info(f"LLMManager initialized with model: {model_path}")
    
    def load_model(self) -> None:
        """
        Load the Llama 3 model into memory.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please download a quantized Llama 3 GGUF model from:\n"
                f"https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF\n"
                f"Recommended: Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
            )
        
        logger.info(f"Loading model: {self.model_path}")
        logger.info(f"Model size: {self.model_path.stat().st_size / (1024**3):.2f} GB")
        
        # Determine if GPU offloading is enabled
        if self.gpu_layers > 0:
            logger.info(f"GPU offloading enabled: {self.gpu_layers} layers")
        else:
            logger.info("Running on CPU only")
        
        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                n_threads=self.threads,
                n_gpu_layers=self.gpu_layers,
                n_batch=512,
                use_mmap=True,      # Memory map the model file
                use_mlock=False,    # Don't lock model in RAM
                verbose=False,
                embedding=False     # We use separate embedding model
            )
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def generate(self, 
                 prompt: str,
                 temperature: float = 0.3,
                 top_p: float = 0.9,
                 max_tokens: int = 512,
                 stop_sequences: list = None) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0 = deterministic, 1 = creative)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop_sequences: List of sequences to stop generation at
            
        Returns:
            Generated text response
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Generating response with temperature={temperature}, max_tokens={max_tokens}")
        
        try:
            response = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences or [],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            logger.debug(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_stream(self,
                        prompt: str,
                        temperature: float = 0.3,
                        top_p: float = 0.9,
                        max_tokens: int = 512,
                        stop_sequences: list = None) -> Generator[str, None, None]:
        """
        Generate a streaming response from the model.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            stop_sequences: List of sequences to stop generation at
            
        Yields:
            Chunks of generated text
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Streaming generation with temperature={temperature}")
        
        try:
            for token in self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences or [],
                echo=False,
                stream=True
            ):
                yield token['choices'][0]['text']
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
    
    def format_prompt(self, 
                      system_prompt: str,
                      user_question: str,
                      context: str = "") -> str:
        """
        Format a prompt for Llama 3 Instruct model.
        
        Args:
            system_prompt: System instructions
            user_question: User's question
            context: Retrieved context from vector database
            
        Returns:
            Formatted prompt string
        """
        # Llama 3 Instruct format
        if context:
            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"Context from documents:\n{context}\n\n"
                f"Question: {user_question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        
        return full_prompt
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded():
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "context_size": self.context_size,
            "gpu_layers": self.gpu_layers,
            "threads": self.threads,
            "model_size_gb": self.model_path.stat().st_size / (1024**3)
        }
    
    def unload_model(self) -> None:
        """
        Unload the model from memory.
        """
        if self.model is not None:
            logger.info("Unloading model from memory")
            del self.model
            self.model = None
            logger.info("Model unloaded")
