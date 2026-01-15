# BitCore-Demos

Interactive demonstrations showcasing the performance and capabilities of [BitCore](https://github.com/olivergrainge/BitCore) - a library for accelerating Ternary Quantized models on various backend with optimized inference.

## Overview

### What is BitNet?

The increasing size of large language models has posed challenges for deployment and raised concerns about environmental impact due to high energy consumption. Microsoft Research introduced BitNet, a scalable and stable 1-bit Transformer architecture designed for large language models. Specifically, they introduced BitLinear as a drop-in replacement of the `nn.Linear` layer in order to train 1-bit weights from scratch. Experimental results on language modeling show that BitNet achieves competitive performance while substantially reducing memory footprint and energy consumption, compared to state-of-the-art 8-bit quantization methods and FP16 Transformer baselines. Furthermore, BitNet exhibits a scaling law akin to full-precision Transformers, suggesting its potential for effective scaling to even larger language models while maintaining efficiency and performance benefits.

*BitNet was originally developed by Microsoft Research. For more information, see the [official BitNet repository](https://github.com/microsoft/BitNet).*

### BitCore: ARM Acceleration for BitNet

[BitCore](https://github.com/olivergrainge/BitCore) is a library that accelerates ternary models, like BitNet models on ARM hardware. This repository contains demos that highlight the practical benefits of using BitCore's optimized inference backend, which achieves significant speedups over standard PyTorch implementations. 

**Performance on ARM M4 Mac:**
- **BitCore Backend (BitOps)**: Achieves **12 tokens/second** with compressed 2-bit weight representations
- **PyTorch Native (FP32)**: Achieves approximately **0.5 tokens/second** with full-precision weights

The BitCore backend uses compressed 2-bit weight representations, resulting in **80% memory savings** compared to standard FP32 implementations, while delivering **24x faster inference** on ARM M4 hardware. This dramatic improvement makes BitNet models practical for deployment on resource-constrained devices and edge computing scenarios, enabled by BitCore's ARM-optimized acceleration.

## Demos

### BitNet Chat Interface

An interactive Gradio-based web application that demonstrates real-time chat with BitNet models using different inference backends.

**Location:** `bitnet/app.py`

#### Features

- 🤖 **Interactive Chat Interface** - Natural language conversations with BitNet models
- ⚡ **Backend Comparison** - Switch between Pytorch FP32 and BitOps 2-bit compressed backends
- 📊 **Real-time Metrics** - Monitor tokens/sec, memory usage, and generation time
- 🎯 **Streaming Responses** - See model outputs as they're generated
- 💾 **Conversation History** - Maintain context across multiple turns
- 🎨 **Modern UI** - Clean, intuitive Gradio interface

#### Performance Comparison

##### Pytorch FP32 Backend (Baseline)
<img src="assets/baseline.gif" alt="Pytorch FP32 Backend" width="800">

*Standard PyTorch inference with FP32 precision*

##### BitOps Backend (Optimized)
<img src="assets/bitops.gif" alt="BitOps Backend" width="800">

*Accelerated inference using BitOps with ternary quantization*

The BitOps backend demonstrates **significant speedups** over the baseline FP32 implementation while maintaining model quality through quantization-aware training. On ARM M4 Mac hardware, the BitCore backend achieves **12 tokens/second** compared to **0.5 tokens/second** with PyTorch native FP32, representing a **24x speedup**.

## Installation

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/olivergrainge/BitCore-Demos.git
cd BitCore-Demos
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install BitCore:
```bash
pip install git+https://github.com/olivergrainge/BitCore.git
```

4. Install BitOps for accelerated inference:
```bash
pip install git+https://github.com/OliverGrainge/BitOps.git
```

## Usage

### Running the BitNet Chat Interface

```bash
cd bitnet
python app.py
```

The web interface will launch at `http://localhost:7860` by default.

#### Command Line Arguments

```bash
python app.py --host 0.0.0.0 --port 7860 --share
```

**Options:**
- `--host`: Host to bind the server to (default: `0.0.0.0`)
- `--port`: Port to run the server on (default: `7860`)
- `--share`: Create a public link for sharing

### Using the Interface

1. **Select Backend**: Choose between "Pytorch FP32" or "BitOps Backend"
2. **Load Model**: Click "Load Model" to initialize the BitNet model
3. **Chat**: Type your message and press Enter or click Send
4. **Monitor Performance**: Watch real-time metrics update as responses generate
5. **Compare Backends**: Switch backends and compare performance metrics

## Performance Metrics

The interface displays several key performance indicators:

- **Generation Speed**: Tokens generated per second
- **Tokens Generated**: Total number of tokens in the response
- **Time Elapsed**: Total time for response generation
- **Model Memory Usage**: Memory footprint of the loaded model

### Typical Performance Gains

When using the BitOps backend vs. Pytorch FP32 (measured on ARM M4 Mac):

- ⚡ **24x faster** token generation (12 tokens/sec vs 0.5 tokens/sec)
- 💾 **80% memory savings** through compressed 2-bit weight representations
- 🎯 **Maintained accuracy** via quantization-aware training

*Performance on ARM M4 Mac: BitCore backend achieves 12 tokens/second with 2-bit compressed weights, while PyTorch native FP32 achieves approximately 0.5 tokens/second. Actual performance may vary based on hardware, model size, and sequence length.*

## Model Information

The demo uses the `microsoft/bitnet-b1.58-2B-4T-bf16` model, which features:

- **1.58-bit weights** (ternary: -1, 0, 1)
- **2B parameters**
- **Quantization-aware training**
- **Chat-optimized** for conversational tasks

## Technical Details

### Architecture

The application uses:

- **BitCore**: Quantization-aware ternary linear layers
- **BitOps**: (Optional) Optimized CUDA kernels for ternary operations
- **Transformers**: Model loading and tokenization
- **Gradio**: Web interface framework


**Note:** BitCore and BitOps must be installed separately from their GitHub repositories (see Installation section above).

## Related Projects

- **[BitCore](https://github.com/olivergrainge/BitCore)**: Quantization-aware ternary linear layers
- **[BitOps](https://github.com/OliverGrainge/BitOps)**: Low-level bitwise operations for accelerated inference

## License

MIT License - see LICENSE file for details

## Author

Oliver Grainge

## Contributing

Contributions are welcome! Feel free to:

- Add new demos showcasing BitCore capabilities
- Improve existing demonstrations
- Optimize performance
- Enhance documentation

Please submit a Pull Request with your changes.

## Citation

If you use BitCore or these demos in your research, please cite:

```bibtex
@software{bitcore2024,
  author = {Grainge, Oliver},
  title = {BitCore: Quantization-Aware Ternary Neural Networks},
  year = {2024},
  url = {https://github.com/olivergrainge/BitCore}
}
```

## Acknowledgments

- **[Microsoft BitNet](https://github.com/microsoft/BitNet)** - Original BitNet architecture and research. This project builds upon the foundational work by Microsoft Research on 1-bit Transformer architectures.
- The PyTorch team for the excellent deep learning framework
- Gradio for the user-friendly web interface framework