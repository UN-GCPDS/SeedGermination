# SeedGermination

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?logo=apache&logoColor=white)
![Python](https://img.shields.io/badge/python-3.11-blue.svg?logo=python&logoColor=white)
![torch 2.6.0](https://img.shields.io/badge/torch-2.6.0-blue.svg?logo=pytorch&logoColor=white)
![ai-edge-torch](https://img.shields.io/badge/ai--edge--torch-0.4.0-blue)
![ai_edge_litert](https://img.shields.io/badge/ai__edge__litert-1.2.0-blue)
![gcpds-cv-pykit](https://img.shields.io/badge/gcpds--cv--pykit-0.1.0.70-blue)

A repository for seed germination detection and segmentation using deep learning models (UNet, ResUNet, DeepLabV3), optimized for deployment on Raspberry Pi 4 using LiteRT (TensorFlow Lite).

## Table of Contents
- [Installation](#installation)
- [Notebooks](#notebooks)
- [Training](#training)
- [Weights](#weights)
- [Testing on RP4](#testing-on-rp4)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

Recommended Python version: 3.11

Install the required dependencies:

```bash
pip install torch wandb gcpds-cv-pykit ai-edge-torch==0.4.0 numpy matplotlib psutil
```

For Raspberry Pi 4 deployment, you will need `ai-edge-litert`.

## Notebooks

The `Notebooks` directory contains source code for training, testing, and exporting models.

### Training
Located in `Notebooks/Training/`:
- `unet-seeds.ipynb`: Training process for the U-Net architecture.
- `resunet-seeds.ipynb`: Training process for the ResUNet architecture.
- `deeplabv3-seeds.ipynb`: Training process for the DeepLabV3 architecture.
- `RESULTS.md`: Detailed training description and segmentation performance tables for all evaluated models and loss functions.

To run these notebooks:
```bash
jupyter lab Notebooks/Training/unet-seeds.ipynb
```

### Model test
Located in `Notebooks/Model test/`:
- `inferece-models.ipynb`: Notebook for running inference tests and evaluating model performance using `wandb` and `gcpds-cv-pykit`.

### Export Pytorch to LiteRT
Located in `Notebooks/Export Pytorch to LiteRT/`:
- `unet-to-litert-seeds.ipynb`: Converts the PyTorch U-Net model to LiteRT format using `ai-edge-torch`.

## Weights

The `Weights` directory contains the optimized model ready for deployment.

- `Weights/mobilenetv3_unet_dynamic.tflite`: A MobileNetV3-based U-Net model with dynamic quantization, exported for LiteRT.

## Testing on RP4

The `Testing on RP4` directory contains scripts to benchmark the model on a Raspberry Pi 4.

**Requirements for RP4:**
- Python 3.11 (recommended)
- `ai_edge_litert`
- `numpy`, `psutil`, `matplotlib`

**Running the benchmark:**

```bash
python "Testing on RP4/rpi4_benchmark.py" --model Weights/mobilenetv3_unet_dynamic.tflite
```

This script measures warm-up performance, CPU/memory usage, average inference time, and throughput.

## Examples

### Inference with LiteRT

Below is an example of how to load the `.tflite` model and perform inference (based on `rpi4_benchmark.py` usage):

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Load the model
model_path = "Weights/mobilenetv3_unet_dynamic.tflite"
interpreter = Interpreter(model_path=model_path, num_threads=4)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data (using random data as example)
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Inference finished. Output shape:", output_data.shape)
```

### Running the RPi4 Benchmark

To execute the benchmark script provided in the repository:

```bash
# From the repository root
python "Testing on RP4/rpi4_benchmark.py" --model Weights/mobilenetv3_unet_dynamic.tflite --num_runs 100
```

## Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewFeature`).
3.  Commit your changes (`git commit -m 'Add NewFeature'`).
4.  Push to the branch (`git push origin feature/NewFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the Apache 2 License - see the [LICENSE](LICENSE) file for details.
