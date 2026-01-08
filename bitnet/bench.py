#!/usr/bin/env python3
"""
Simplified benchmarking script for BitNet model with custom BitLinear layers.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import time
import psutil
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
from bitcore import BitLinear
from tqdm import tqdm
import argparse
import json


def get_memory_mb():
    """Get current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def replace_bitnet_layers(model, quant_type="bitnet"):
    """Replace AutoBitLinear layers with custom BitLinear."""
    layers = [(name, m) for name, m in model.named_modules() 
              if m.__class__.__name__ == "AutoBitLinear"]
    
    for name, module in tqdm(layers, desc="Quantizing"):
        custom_layer = BitLinear.from_linear(module, quant_type=quant_type)
        custom_layer.eval()
        
        parts = name.split('.')
        parent = model.get_submodule('.'.join(parts[:-1])) if len(parts) > 1 else model
        setattr(parent, parts[-1], custom_layer)
    
    return len(layers)


def deploy_bitlinear_layers(model):
    """Deploy BitLinear layers for optimized inference."""
    layers = [(name, m) for name, m in model.named_modules() 
              if isinstance(m, BitLinear) and not m._is_deployed]
    
    for _, module in tqdm(layers, desc="Deploying"):
        module.deploy()
    
    return len(layers)


def run_benchmark(
    model_id,
    device="auto",
    quant_type="bitnet",
    deploy=False,
    prompt="Explain quantum computing in simple terms.",
    max_new_tokens=256,
    num_runs=3,
    warmup_runs=1,
):
    """Run benchmark and return results."""
    
    print("=" * 80)
    print(f"BITNET BENCHMARK: {model_id}")
    print(f"Quantization: {quant_type} | Deploy: {deploy} | Tokens: {max_new_tokens}")
    print("=" * 80 + "\n")
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Track memory
    mem_before = get_memory_mb()
    
    # Load model
    print("Loading model...")
    transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map=device if device != "cpu" else None,
    )
    transformers_logging.set_verbosity_warning()
    model.eval()
    mem_after_load = get_memory_mb()
    print(f"✓ Loaded ({mem_after_load - mem_before:.0f} MB)\n")
    
    # Quantize
    print(f"Quantizing ({quant_type})...")
    layers_replaced = replace_bitnet_layers(model, quant_type)
    mem_after_quant = get_memory_mb()
    print(f"✓ Quantized {layers_replaced} layers\n")
    
    # Deploy if requested
    layers_deployed = 0
    if deploy:
        print("Deploying...")
        layers_deployed = deploy_bitlinear_layers(model)
        print(f"✓ Deployed {layers_deployed} layers\n")
    
    model_size = get_model_size_mb(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.1f}M params, {model_size:.1f} MB\n")
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs['input_ids'].shape[-1]
    
    # Warmup
    if warmup_runs > 0:
        print(f"Warmup ({warmup_runs} run(s))...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=max_new_tokens, 
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
            if device == "cuda":
                torch.cuda.synchronize()
        print("✓ Warmup complete\n")
    
    # Benchmark
    print(f"Benchmarking ({num_runs} run(s))...")
    ttfts, total_times, token_counts = [], [], []
    
    for run in range(num_runs):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Measure TTFT
            ttft_start = time.time()
            model.generate(**inputs, max_new_tokens=1, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
            if device == "cuda":
                torch.cuda.synchronize()
            ttft = time.time() - ttft_start
            
            # Full generation
            gen_start = time.time()
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
            if device == "cuda":
                torch.cuda.synchronize()
            gen_time = time.time() - gen_start
        
        tokens = outputs.shape[-1] - input_len
        ttfts.append(ttft)
        total_times.append(gen_time)
        token_counts.append(tokens)
        
        print(f"  Run {run+1}: TTFT={ttft*1000:.1f}ms | "
              f"Time={gen_time:.2f}s | Tokens={tokens} | "
              f"Speed={tokens/gen_time:.1f} tok/s")
    
    # Calculate averages
    avg_ttft = sum(ttfts) / len(ttfts)
    avg_time = sum(total_times) / len(total_times)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_speed = avg_tokens / avg_time
    
    # Results
    results = {
        "model": model_id,
        "device": device,
        "quant_type": quant_type,
        "deployed": deploy,
        "params_m": total_params / 1e6,
        "model_size_mb": model_size,
        "layers_replaced": layers_replaced,
        "layers_deployed": layers_deployed,
        "ttft_ms": avg_ttft * 1000,
        "gen_time_s": avg_time,
        "tokens_generated": int(avg_tokens),
        "tokens_per_sec": avg_speed,
        "ms_per_token": (avg_time / avg_tokens) * 1000,
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "mem_after_quant_mb": mem_after_quant,
    }
    
    if torch.cuda.is_available():
        results["gpu_mem_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
    
    return results


def print_results(results):
    """Print results in clean format."""
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n📊 Model: {results['model']}")
    print(f"   {results['params_m']:.1f}M params | {results['model_size_mb']:.1f} MB | "
          f"{results['layers_replaced']} layers quantized")
    
    print(f"\n⚡ Performance:")
    print(f"   TTFT: {results['ttft_ms']:.1f} ms")
    print(f"   Speed: {results['tokens_per_sec']:.1f} tokens/s")
    print(f"   Time per token: {results['ms_per_token']:.1f} ms")
    
    print(f"\n💾 Memory:")
    print(f"   Model size: {results['model_size_mb']:.1f} MB")
    print(f"   After quantization: {results['mem_after_quant_mb']:.1f} MB")
    if "gpu_mem_mb" in results:
        print(f"   GPU allocated: {results['gpu_mem_mb']:.1f} MB")
    
    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BitNet model")
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quant-type", default="bitnet")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    results = run_benchmark(
        model_id=args.model,
        device=args.device,
        quant_type=args.quant_type,
        deploy=args.deploy,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
        warmup_runs=args.warmup,
    )
    
    print_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved to {args.output}\n")


if __name__ == "__main__":
    main()