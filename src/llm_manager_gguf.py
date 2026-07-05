"""
LLM Integration Module for Pistos.ai (llama-cpp-python Version)
Handles Llama 3 inference using llama-cpp-python for GGUF models.
Uses pre-built wheels - no compilation required.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Generator

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages Llama 3 inference using llama-cpp-python for GGUF models.
    """
    
    def __init__(self,
                 model_path: str = "data/models/llama-3-8b-instruct.Q5_K_M.gguf",
                 context_size: int = 4096,
                 gpu_layers: int = 0,
                 threads: int = 4,
                 model_type: str = "llama"):
        """
        Initialize the LLM manager.
        
        Args:
            model_path: Path to the GGUF model file
            context_size: Maximum context window size
            gpu_layers: Number of layers to offload to GPU (0 for CPU)
            threads: Number of CPU threads for inference
            model_type: Not used with llama-cpp-python (kept for compatibility)
        """
        self.model_path = Path(model_path)
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.threads = threads
        
        self.model = None
        
        logger.info(f"LLMManager initialized with GGUF model: {model_path}")
    
    def load_model(self) -> None:
        """
        Load the GGUF model into memory.
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. Please install it:\n"
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please download a GGUF model from:\n"
                f"https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF"
            )
        
        logger.info(f"Loading GGUF model: {self.model_path}")
        logger.info(f"Model size: {self.model_path.stat().st_size / (1024**3):.2f} GB")

        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                n_threads=self.threads,
                n_gpu_layers=self.gpu_layers,
                n_batch=256,
                n_ubatch=256,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
                logits_all=False,
                embedding=False
            )
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        return self.model is not None
    
    def generate(self, 
                 prompt: str,
                 temperature: float = 0.3,
                 top_p: float = 0.9,
                 max_tokens: int = 512,
                 stop_sequences: list = None) -> str:
        """Generate a response from the model."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Generating response with temperature={temperature}, max_tokens={max_tokens}")
        
        try:
            response = self.model(
                prompt,
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
        """Generate a streaming response from the model."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Streaming generation with temperature={temperature}")
        
        try:
            for token in self.model(
                prompt,
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
        """Format a prompt for Llama 3 Instruct model."""
        if context:
            full_prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"Context from documents:\n{context}\n\n"
                f"Question: {user_question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            full_prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        return full_prompt
    
    def get_model_info(self) -> dict:
        if not self.is_loaded():
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "context_size": self.context_size,
            "gpu_layers": self.gpu_layers,
            "threads": self.threads,
            "model_size_gb": self.model_path.stat().st_size / (1024**3) if self.model_path.exists() else 0
        }
    
    def unload_model(self) -> None:
        if self.model is not None:
            logger.info("Unloading model from memory")
            del self.model
            self.model = None
            logger.info("Model unloaded")
