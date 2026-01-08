#!/usr/bin/env python3
"""
Interactive chat interface for BitNet model with custom BitLinear layers.

This script:
1. Loads the BitNet model
2. Replaces all AutoBitLinear layers with custom BitLinear implementation
3. Provides an interactive chat interface with streaming output
4. Optionally deploys layers for optimized inference
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
from transformers import TextIteratorStreamer
from typing import Optional, List, Dict
import sys
from bitcore import BitLinear
from tqdm import tqdm
from threading import Thread


def replace_bitnet_layers(model, quant_type: str = "bitnet", verbose: bool = True):
    """
    Replace all AutoBitLinear layers with custom BitLinear implementation.
    
    Args:
        model: The model to modify
        quant_type: Quantization type for BitLinear layers
        verbose: Whether to print replacement progress
    
    Returns:
        int: Number of layers replaced
    """
    # First, collect all layers that need to be replaced
    layers_to_replace = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "AutoBitLinear":
            layers_to_replace.append((name, module))
    
    # Now replace them with tqdm progress bar
    replaced_count = 0
    for name, module in tqdm(layers_to_replace, desc="Quantizing layers", disable=not verbose):
        # Create custom BitLinear layer from the original
        custom_layer = BitLinear.from_linear(module, quant_type=quant_type)
        custom_layer.eval()
        
        # Replace in model hierarchy
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        
        setattr(parent, child_name, custom_layer)
        replaced_count += 1
    
    return replaced_count


def deploy_bitlinear_layers(model, verbose: bool = True):
    """
    Deploy all BitLinear layers for optimized inference.
    
    Args:
        model: The model containing BitLinear layers
        verbose: Whether to print deployment progress
    
    Returns:
        int: Number of layers deployed
    """
    # First, collect all layers that need to be deployed
    layers_to_deploy = []
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and not module._is_deployed:
            layers_to_deploy.append((name, module))
    
    # Now deploy them with tqdm progress bar
    deployed_count = 0
    for name, module in tqdm(layers_to_deploy, desc="Deploying layers", disable=not verbose):
        module.deploy()
        deployed_count += 1
    
    return deployed_count


class ChatBot:
    """Interactive chatbot with custom BitLinear layers and streaming output."""
    
    def __init__(
        self,
        model_id: str = "microsoft/bitnet-b1.58-2B-4T-bf16",
        device: str = "auto",
        quant_type: str = "bitnet",
        deploy: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
    ):
        """
        Initialize the chatbot.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ("cpu", "cuda", or "auto")
            quant_type: Quantization type for BitLinear layers
            deploy: Whether to deploy layers for optimized inference
            max_new_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream output token-by-token
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream
        self.conversation_history: List[Dict[str, str]] = []
        self.past_key_values = None  # KV cache for efficient multi-turn conversations
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model: {model_id}")
        print(f"Device: {self.device}\n")
        
        # Load tokenizer
        print("1. Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("   ✓ Tokenizer loaded\n")
        
        # Load model
        print("2. Loading model...")
        # Suppress transformers warnings during model loading
        transformers_logging.set_verbosity_error()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            device_map=self.device if self.device != "cpu" else None,
        )
        transformers_logging.set_verbosity_warning()  # Restore default logging
        self.model.eval()
        print("   ✓ Model loaded\n")
        
        # Replace layers
        print(f"3. Quantizing layers ({quant_type})...")
        replaced = replace_bitnet_layers(self.model, quant_type=quant_type)
        print(f"\n   ✓ Quantized {replaced} layers\n")
        
        # Deploy if requested
        if deploy:
            print("4. Deploying layers for optimized inference...")
            deployed = deploy_bitlinear_layers(self.model)
            print(f"   ✓ Deployed {deployed} layers\n")
        
        print("=" * 80)
        print("Chatbot ready! Type 'quit', 'exit', or 'q' to end the conversation.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'help' for more commands.")
        if self.stream:
            print("Streaming mode: ON (text will appear token-by-token)")
        else:
            print("Streaming mode: OFF")
        print("=" * 80 + "\n")
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear conversation history and KV cache."""
        self.conversation_history = []
        self.past_key_values = None
        print("✓ Conversation history cleared.\n")
    
    def toggle_streaming(self):
        """Toggle streaming mode on/off."""
        self.stream = not self.stream
        status = "ON" if self.stream else "OFF"
        print(f"✓ Streaming mode: {status}\n")
    
    def generate_response(self, user_message: str) -> str:
        """
        Generate a response to the user's message.
        
        Args:
            user_message: The user's input message
            
        Returns:
            The model's response
        """
        # Add user message to history
        self.add_message("user", user_message)
        
        # Prepare prompt with conversation history
        prompt = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if self.stream:
            # Streaming generation
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the output
            response = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                response += new_text
            
            thread.join()
            
        else:
            # Non-streaming generation (original behavior)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            )
        
        # Add assistant response to history
        self.add_message("assistant", response)
        
        return response
    
    def show_help(self):
        """Display help message."""
        print("\n" + "=" * 80)
        print("COMMANDS:")
        print("=" * 80)
        print("  quit, exit, q     - Exit the chatbot")
        print("  clear             - Clear conversation history")
        print("  help              - Show this help message")
        print("  history           - Show conversation history")
        print("  settings          - Show current generation settings")
        print("  stream            - Toggle streaming mode on/off")
        print("=" * 80 + "\n")
    
    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("\n(No conversation history yet)\n")
            return
        
        print("\n" + "=" * 80)
        print("CONVERSATION HISTORY:")
        print("=" * 80)
        for i, msg in enumerate(self.conversation_history, 1):
            role = msg["role"].upper()
            content = msg["content"]
            print(f"\n[{i}] {role}:")
            print(f"{content}")
        print("\n" + "=" * 80 + "\n")
    
    def show_settings(self):
        """Display current generation settings."""
        print("\n" + "=" * 80)
        print("CURRENT SETTINGS:")
        print("=" * 80)
        print(f"  Model: {self.model_id}")
        print(f"  Device: {self.device}")
        print(f"  Max new tokens: {self.max_new_tokens}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-p: {self.top_p}")
        print(f"  Streaming: {'ON' if self.stream else 'OFF'}")
        print("=" * 80 + "\n")
    
    def chat(self):
        """Run the interactive chat loop."""
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋\n")
                    break
                
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                elif user_input.lower() == 'settings':
                    self.show_settings()
                    continue
                
                elif user_input.lower() == 'stream':
                    self.toggle_streaming()
                    continue
                
                # Generate response
                print("\nAssistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                
                # Only print response if not streaming (already printed during streaming)
                if not self.stream:
                    print(response, end="")
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋\n")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chat with BitNet model using custom BitLinear layers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy layers for optimized inference"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 for greedy decoding)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt for the conversation"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (wait for complete response)"
    )
    
    args = parser.parse_args()
    
    # Create chatbot
    chatbot = ChatBot(
        model_id=args.model,
        device=args.device,
        deploy=args.deploy,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=not args.no_stream,
    )
    
    # Add system prompt if provided
    if args.system_prompt:
        chatbot.add_message("system", args.system_prompt)
    
    # Start chat loop
    chatbot.chat()


if __name__ == "__main__":
    main()