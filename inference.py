#!/usr/bin/env python3
"""
DeepSeek-R1 Inference Script

A simple, user-friendly script for running inference with DeepSeek-R1 models.
Supports multiple backends: Transformers, Unsloth, llama.cpp, and Ollama.

Usage Examples:
    # Interactive chat with downloaded model
    python inference.py --interactive
    
    # Single prompt with Ollama
    python inference.py --backend ollama --model "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_XL" --prompt "Explain quantum computing"
    
    # Local model inference
    python inference.py --backend transformers --model "./cache/unsloth/DeepSeek-R1-0528-Qwen3-8B" --prompt "Hello world"

Author: DeepSeeks-Trial Project
"""

import argparse
import os
import sys
import time
from typing import Optional, Dict, Any

# Backend imports (with error handling for optional dependencies)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import subprocess
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False

# DeepSeek model presets
DEEPSEEK_PRESETS = {
    "qwen3-8b-gguf": {
        "repo_id": "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        "ollama_model": "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_XL",
        "description": "DeepSeek-R1-0528-Qwen3-8B (Q4_K_XL quantization, ~5GB)"
    },
    "qwen3-8b-q2": {
        "repo_id": "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        "patterns": ["*UD-Q2_K_XL*"],
        "description": "DeepSeek-R1-0528-Qwen3-8B (Dynamic 2.7-bit quantization, ~3GB)"
    },
    "full-r1-q2": {
        "repo_id": "unsloth/DeepSeek-R1-0528-GGUF", 
        "patterns": ["*UD-Q2_K_XL*"],
        "description": "DeepSeek-R1-0528 Full (Dynamic 2.7-bit quantization, ~251GB)"
    },
    "full-r1-1bit": {
        "repo_id": "unsloth/DeepSeek-R1-0528-GGUF",
        "patterns": ["*UD-IQ1_S*"], 
        "description": "DeepSeek-R1-0528 Full (Dynamic 1.78-bit quantization, ~185GB)"
    }
}

class DeepSeekInference:
    """Main inference class supporting multiple backends."""
    
    def __init__(self, backend: str, model_path: str, **kwargs):
        self.backend = backend
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        
        # DeepSeek recommended settings
        self.default_generation_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
            "max_new_tokens": 512,
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load model based on the specified backend."""
        print(f"Loading model using {self.backend} backend...")
        
        if self.backend == "transformers":
            self._load_transformers_model()
        elif self.backend == "unsloth":
            self._load_unsloth_model()
        elif self.backend == "ollama":
            self._setup_ollama()
        elif self.backend == "llama.cpp":
            self._setup_llama_cpp()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _load_transformers_model(self):
        """Load model using Transformers library."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Install with: pip install transformers torch")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"Model loaded: {self.model_path}")
    
    def _load_unsloth_model(self):
        """Load model using Unsloth FastLanguageModel."""
        if not UNSLOTH_AVAILABLE:
            raise ImportError("unsloth library not available. Install following Unsloth documentation.")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.kwargs.get("max_seq_length", 4096),
            dtype=torch.bfloat16,
            load_in_4bit=self.kwargs.get("load_in_4bit", True),
        )
        print(f"Unsloth model loaded: {self.model_path}")
    
    def _setup_ollama(self):
        """Setup for Ollama backend."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama library not available. Install with: pip install ollama")
        
        # Test if Ollama is running and model is available
        try:
            models = ollama.list()
            available_models = [m['name'] for m in models.get('models', [])]
            if self.model_path not in available_models:
                print(f"Model '{self.model_path}' not found in Ollama. Available models: {available_models}")
                print(f"To pull the model, run: ollama pull {self.model_path}")
                sys.exit(1)
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running (ollama serve)")
            sys.exit(1)
        
        print(f"Ollama model ready: {self.model_path}")
    
    def _setup_llama_cpp(self):
        """Setup for llama.cpp backend."""
        if not SUBPROCESS_AVAILABLE:
            raise ImportError("subprocess not available")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Find llama.cpp binary (user should have it in PATH or specify)
        llama_cpp_path = self.kwargs.get("llama_cpp_path", "llama-cli")
        try:
            subprocess.run([llama_cpp_path, "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"llama.cpp binary not found: {llama_cpp_path}")
            print("Please install llama.cpp and ensure 'llama-cli' is in your PATH, or specify --llama_cpp_path")
            sys.exit(1)
        
        self.llama_cpp_path = llama_cpp_path
        print(f"llama.cpp setup complete: {self.model_path}")
    
    def format_prompt(self, user_message: str) -> str:
        """Format user message according to DeepSeek chat format."""
        if not user_message.startswith(""):
            user_message = f"{user_message}"
        return user_message
    
    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate response for the given prompt."""
        # Merge generation config
        config = {**self.default_generation_config, **generation_kwargs}
        
        # Format prompt
        formatted_prompt = self.format_prompt(prompt)
        
        if self.backend in ["transformers", "unsloth"]:
            return self._generate_transformers(formatted_prompt, config)
        elif self.backend == "ollama":
            return self._generate_ollama(formatted_prompt, config)
        elif self.backend == "llama.cpp":
            return self._generate_llama_cpp(formatted_prompt, config)
    
    def _generate_transformers(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate using Transformers/Unsloth."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=config["temperature"],
                top_p=config["top_p"],
                do_sample=config["do_sample"],
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _generate_ollama(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate using Ollama."""
        try:
            response = ollama.generate(
                model=self.model_path,
                prompt=prompt,
                options={
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "num_predict": config["max_new_tokens"],
                }
            )
            return response['response'].strip()
        except Exception as e:
            return f"Error from Ollama: {e}"
    
    def _generate_llama_cpp(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate using llama.cpp."""
        try:
            cmd = [
                self.llama_cpp_path,
                "--model", self.model_path,
                "--prompt", prompt,
                "--temp", str(config["temperature"]),
                "--top-p", str(config["top_p"]),
                "--n-predict", str(config["max_new_tokens"]),
                "--silent-prompt",  # Don't echo the prompt
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error from llama.cpp: {e.stderr}"
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("=== DeepSeek-R1 Interactive Chat ===")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        print("Type 'clear' to clear the conversation history.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 User: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'clear':
                    print("Conversation history cleared.")
                    continue
                
                if not user_input:
                    continue
                
                print("\n🤖 Assistant: ", end="", flush=True)
                
                # Generate response
                response = self.generate(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}")

def list_presets():
    """Display available model presets."""
    print("Available DeepSeek model presets:")
    print("-" * 50)
    for name, info in DEEPSEEK_PRESETS.items():
        print(f"  {name:15} - {info['description']}")
    print("\nUse with: --preset <preset_name>")

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-R1 Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat with local model
  python inference.py --backend transformers --model ./cache/unsloth/DeepSeek-R1-0528-Qwen3-8B --interactive
  
  # Single prompt with Ollama
  python inference.py --backend ollama --model "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_XL" --prompt "Hello"
  
  # Use preset model
  python inference.py --preset qwen3-8b-gguf --backend ollama --interactive
  
  # llama.cpp inference
  python inference.py --backend llama.cpp --model ./model.gguf --prompt "Explain AI"
        """
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Path to model or Ollama model name")
    model_group.add_argument("--preset", type=str, choices=list(DEEPSEEK_PRESETS.keys()), help="Use a predefined model preset")
    
    # Backend selection
    parser.add_argument("--backend", type=str, choices=["transformers", "unsloth", "ollama", "llama.cpp"], 
                        default="transformers", help="Inference backend to use")
    
    # Generation options
    parser.add_argument("--prompt", type=str, help="Single prompt for inference (instead of interactive mode)")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.6)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling (default: 0.95)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    
    # Backend-specific options
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length (for Unsloth)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load in 4-bit (for Unsloth)")
    parser.add_argument("--llama_cpp_path", type=str, default="llama-cli", help="Path to llama.cpp binary")
    
    # Utility options
    parser.add_argument("--list-presets", action="store_true", help="List available model presets")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_presets:
        list_presets()
        return
    
    # Validate arguments
    if not args.model and not args.preset:
        print("Error: Either --model or --preset must be specified")
        print("Use --list-presets to see available presets")
        sys.exit(1)
    
    if not args.prompt and not args.interactive:
        print("Error: Either --prompt or --interactive must be specified")
        sys.exit(1)
    
    # Resolve model path from preset
    if args.preset:
        preset = DEEPSEEK_PRESETS[args.preset]
        if args.backend == "ollama" and "ollama_model" in preset:
            model_path = preset["ollama_model"]
        else:
            # For other backends, assume model is downloaded to cache
            repo_parts = preset["repo_id"].split("/")
            model_path = f"./cache/{'/'.join(repo_parts)}"
        print(f"Using preset '{args.preset}': {preset['description']}")
    else:
        model_path = args.model
    
    # Initialize inference engine
    try:
        inference = DeepSeekInference(
            backend=args.backend,
            model_path=model_path,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            llama_cpp_path=args.llama_cpp_path,
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)
    
    # Run inference
    if args.interactive:
        inference.interactive_chat()
    else:
        print(f"Prompt: {args.prompt}")
        print("-" * 50)
        
        start_time = time.time()
        response = inference.generate(
            args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens
        )
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"\nGeneration time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 