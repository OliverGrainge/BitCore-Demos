#!/usr/bin/env python3
"""
Gradio Web App for BitNet Model Chat Interface

Features:
- Interactive chat interface
- Backend selection (Pytorch FP32 / BitOps)
- Real-time performance metrics (tokens/sec, memory usage)
- Streaming responses
- Conversation history management
- Enhanced metrics visualization with progress bars
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
from transformers import TextIteratorStreamer
import gradio as gr
from typing import List, Dict, Tuple, Optional
import time
import psutil
import gc
from threading import Thread
from tqdm import tqdm
from bitcore import BitLinear


def replace_bitnet_layers(model, quant_type: str = "bitnet", verbose: bool = False):
    """
    Replace all AutoBitLinear layers with custom BitLinear implementation.
    
    Args:
        model: The model to modify
        quant_type: Quantization type for BitLinear layers
        verbose: Whether to print replacement progress
    
    Returns:
        int: Number of layers replaced
    """
    layers_to_replace = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "AutoBitLinear":
            layers_to_replace.append((name, module))
    
    replaced_count = 0
    for name, module in tqdm(layers_to_replace, desc="Quantizing layers", disable=not verbose):
        custom_layer = BitLinear.from_linear(module, quant_type=quant_type)
        custom_layer.eval()
        
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        
        setattr(parent, child_name, custom_layer)
        replaced_count += 1
    
    return replaced_count


def deploy_bitlinear_layers(model, verbose: bool = False):
    """
    Deploy all BitLinear layers for optimized inference.
    
    Args:
        model: The model containing BitLinear layers
        verbose: Whether to print deployment progress
    
    Returns:
        int: Number of layers deployed
    """
    layers_to_deploy = []
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and not module._is_deployed:
            layers_to_deploy.append((name, module))
    
    deployed_count = 0
    for name, module in tqdm(layers_to_deploy, desc="Deploying layers", disable=not verbose):
        module.deploy()
        deployed_count += 1
    
    return deployed_count


def get_model_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_model_size_mb(model):
    """Calculate model static memory size in MB (weights + buffers)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


class BitNetChatApp:
    """Gradio web application for BitNet chat."""
    
    def __init__(self, model_id: str = "microsoft/bitnet-b1.58-2B-4T-bf16"):
        """Initialize the chat application."""
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history: List[Dict[str, str]] = []
        self.current_backend = None
        self.model_memory = 0
        
        # Generation settings
        self.max_new_tokens = 1024
        self.temperature = 0.7
        self.top_p = 0.9
        
        # Performance tracking
        self.max_tokens_per_sec = 100  # For progress bar scaling
    
    def load_model(self, backend: str, progress=gr.Progress()) -> str:
        """
        Load the model with the specified backend.
        
        Args:
            backend: Either "Pytorch FP32" or "BitOps Backend"
            progress: Gradio progress tracker
            
        Returns:
            Status message
        """
        try:
            progress(0, desc="Cleaning up previous model...")
            
            # Clear previous model and tokenizer if they exist
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Load tokenizer
            progress(0.2, desc="Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
            progress(0.4, desc="Loading model...")
            transformers_logging.set_verbosity_error()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float32,
                device_map=self.device if self.device != "cpu" else None,
            )
            transformers_logging.set_verbosity_warning()
            self.model.eval()
            
            # Configure backend
            if backend == "Pytorch FP32":
                progress(0.6, desc="Applying ternary compression...")
                quant_type = "bitnet"
                deploy = False
            else:  # BitOps Backend
                progress(0.6, desc="Configuring BitOps backend...")
                quant_type = "bitnet"
                deploy = True
            
            # Replace layers
            progress(0.7, desc="Quantizing layers...")
            replaced = replace_bitnet_layers(self.model, quant_type=quant_type, verbose=False)
            
            # Deploy if using BitOps
            if deploy:
                progress(0.85, desc="Deploying BitOps layers...")
                deployed = deploy_bitlinear_layers(self.model, verbose=False)
            
            # Calculate model static memory (weights + buffers)
            # This will be lower for BitOps backend after deploy() due to quantization
            self.model_memory = get_model_size_mb(self.model)
            
            self.current_backend = backend
            progress(1.0, desc="Ready!")
            
            return f"✓ Model loaded with {backend}\n✓ Quantized {replaced} layers\n✓ Model memory: {self.model_memory:.2f} MB"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def get_metrics_display(self, tokens_per_sec: float, token_count: int, elapsed: float) -> Tuple[str, str, str, str]:
        """
        Format metrics for display.
        
        Returns:
            Tuple of (backend_name, tokens_per_sec_str, token_count_str, elapsed_str)
        """
        backend_name = self.current_backend if self.current_backend else "Not loaded"
        
        tokens_per_sec_str = f"{tokens_per_sec:.2f} tokens/sec"
        token_count_str = f"{token_count} tokens"
        elapsed_str = f"{elapsed:.2f}s"
        
        return backend_name, tokens_per_sec_str, token_count_str, elapsed_str
    
    def chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ):
        """
        Generate a response to the user's message.
        
        Args:
            message: User's input message
            history: Chat history as list of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Yields:
            Updated history and performance metrics
        """
        if self.model is None:
            yield (
                history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⚠️ Please load a model first!"}],
                "Not loaded",
                "0.00 tokens/sec",
                "0 tokens",
                "0.00s",
                f"{self.model_memory:.1f} MB"
            )
            return
        
        # Update generation settings (use defaults if not provided)
        if max_tokens is not None:
            self.max_new_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        
        # Convert gradio history to conversation format
        self.conversation_history = []
        for msg in history:
            self.conversation_history.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current message
        self.conversation_history.append({"role": "user", "content": message})
        
        # Prepare prompt
        prompt = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Setup streaming
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
        
        # Start generation in separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        
        # Track performance
        start_time = time.time()
        token_count = 0
        response = ""
        
        thread.start()
        
        # Stream the output
        for new_text in streamer:
            response += new_text
            token_count += 1
            
            # Calculate metrics
            elapsed = time.time() - start_time
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
            
            # Update history with new format
            updated_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            
            # Get formatted metrics
            backend_name, tps_str, token_str, elapsed_str = self.get_metrics_display(
                tokens_per_sec, token_count, elapsed
            )
            
            yield (
                updated_history,
                backend_name,
                tps_str,
                token_str,
                elapsed_str,
                f"{self.model_memory:.1f} MB"
            )
        
        thread.join()
        
        # Final metrics
        elapsed = time.time() - start_time
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
        
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        
        backend_name, tps_str, token_str, elapsed_str = self.get_metrics_display(
            tokens_per_sec, token_count, elapsed
        )
        
        yield (
            updated_history,
            backend_name,
            tps_str,
            token_str,
            elapsed_str,
            f"{self.model_memory:.1f} MB"
        )
    
    def clear_history(self) -> Tuple:
        """Clear conversation history."""
        self.conversation_history = []
        return (
            [],
            self.current_backend if self.current_backend else "Not loaded",
            "0.00 tokens/sec",
            "0 tokens",
            "0.00s",
            f"{self.model_memory:.1f} MB" if self.model_memory else "0 MB"
        )


def create_interface():
    """Create and configure the Gradio interface."""
    app = BitNetChatApp()
    
    with gr.Blocks(title="BitNet Chat Interface") as demo:
        gr.Markdown(
            """
            # 🤖 BitNet Chat Interface
            
            Chat with BitNet models using different inference backends and monitor performance in real-time.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model Configuration")
                
                backend_selector = gr.Radio(
                    choices=["Pytorch FP32", "BitOps Backend"],
                    value="Pytorch FP32",
                    label="Inference Backend",
                    info="Choose the inference optimization strategy"
                )
                
                load_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
                load_status = gr.Textbox(
                    label="Model Status",
                    value="Model not loaded",
                    interactive=False,
                    lines=4
                )
                
                gr.Markdown("### 📊 Performance Metrics")
                
                with gr.Group():
                    backend_display = gr.Label(
                        value="Not loaded",
                        label="Active Backend",
                        show_label=True
                    )
                    
                    with gr.Row():
                        tokens_per_sec_display = gr.Textbox(
                            value="0.00 tokens/sec",
                            label="Generation Speed",
                            interactive=False,
                            scale=1
                        )
                        token_count_display = gr.Textbox(
                            value="0 tokens",
                            label="Tokens Generated",
                            interactive=False,
                            scale=1
                        )
                    
                    with gr.Row():
                        elapsed_display = gr.Textbox(
                            value="0.00s",
                            label="Time Elapsed",
                            interactive=False,
                            scale=1
                        )
                        memory_display = gr.Textbox(
                            value="0 MB",
                            label="Model Memory Usage",
                            interactive=False,
                            scale=1
                        )
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat")
                
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")

        
        # Event handlers
        load_btn.click(
            fn=app.load_model,
            inputs=[backend_selector],
            outputs=[load_status]
        )
        
        msg_input.submit(
            fn=app.chat,
            inputs=[msg_input, chatbot],
            outputs=[
                chatbot,
                backend_display,
                tokens_per_sec_display,
                token_count_display,
                elapsed_display,
                memory_display
            ],
            show_progress=True
        ).then(
            lambda: "",
            outputs=[msg_input]
        )
        
        send_btn.click(
            fn=app.chat,
            inputs=[msg_input, chatbot],
            outputs=[
                chatbot,
                backend_display,
                tokens_per_sec_display,
                token_count_display,
                elapsed_display,
                memory_display
            ],
            show_progress=True
        ).then(
            lambda: "",
            outputs=[msg_input]
        )
        
        clear_btn.click(
            fn=app.clear_history,
            outputs=[
                chatbot,
                backend_display,
                tokens_per_sec_display,
                token_count_display,
                elapsed_display,
                memory_display
            ]
        )
    
    return demo


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BitNet Chat Web Interface")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()