"""
LLM Integration Module for Pistos.ai (Transformers Version)
Handles Llama 3 inference using Hugging Face transformers with 4-bit quantization.
No compilation required - uses pre-built wheels.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages Llama 3 inference using Hugging Face transformers.
    Uses 4-bit quantization for efficient CPU/GPU inference.
    """
    
    def __init__(self,
                 model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 context_size: int = 4096,
                 gpu_layers: int = 0,
                 threads: int = 4,
                 use_quantized: bool = True):
        """
        Initialize the LLM manager.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            context_size: Maximum context window size
            gpu_layers: Not used with transformers (kept for compatibility)
            threads: Number of CPU threads for inference
            use_quantized: Use 4-bit quantization for lower memory usage
        """
        self.model_path = model_path
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.use_quantized = use_quantized
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        logger.info(f"LLMManager initialized with model: {model_path}")
    
    def load_model(self) -> None:
        """
        Load the Llama 3 model into memory.
        """
        logger.info(f"Loading model: {self.model_path}")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple MPS")
        else:
            self.device = torch.device("cpu")
            logger.info("Running on CPU")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set quantization config for lower memory usage
            if self.use_quantized and self.device.type == "cpu":
                logger.info("Using 8-bit quantization for CPU inference...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    load_in_8bit=True,
                    trust_remote_code=True
                )
            elif self.use_quantized and self.device.type == "cuda":
                logger.info("Using 4-bit quantization for GPU inference...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    trust_remote_code=True
                )
            else:
                logger.info("Loading full precision model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
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
        return self.model is not None and self.tokenizer is not None
    
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
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and extract only the generated part
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            
            generated_text = generated_text.strip()
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
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Set up generation kwargs
            generation_kwargs = dict(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Run generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield tokens as they're generated
            for token in streamer:
                yield token
            
            thread.join()
                
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
            "device": str(self.device),
            "quantized": self.use_quantized
        }
    
    def unload_model(self) -> None:
        """
        Unload the model from memory.
        """
        if self.model is not None:
            logger.info("Unloading model from memory")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")
