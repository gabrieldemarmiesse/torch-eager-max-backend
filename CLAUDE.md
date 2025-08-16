# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch backend integration that bridges CPU-only PyTorch with Modular's MAX engine to enable GPU computation without CUDA. The project currently supports float32 tensors with `arange` and `add` operations.

## Development Commands

This project uses `uv` for dependency management:

- **Run the main example**: `uv run main.py`
- **Install dependencies**: `uv sync`
- **Add dependencies**: `uv add <package>`

## Architecture

### Core Components

- **`max_backend.py`**: Contains the main backend implementation
  - `MaxDeviceBackend`: Main class that registers the custom device with PyTorch
  - `MaxDeviceModule`: Implements PyTorch device module interface for the "max_device"
  - `register_max_ops()`: Registers custom operations (`arange`, `add`) for the PrivateUse1 dispatch key

- **`main.py`**: Example demonstrating the backend usage with tensor operations

### Integration Flow

1. **Backend Registration**: `MaxDeviceBackend.register()` renames PrivateUse1 to "max_device" and registers device module
2. **Operation Dispatch**: Custom operations are implemented using MAX graph operations and executed via InferenceSession
3. **Tensor Creation**: Tensors created on "max_device" are executed using MAX engine but maintain PyTorch tensor interface

### Dependencies

- `torch-max-backend>=0.1.1`: Modular MAX backend for PyTorch
- `torch`: PyTorch framework (CPU-only version from pytorch-cpu index)
- MAX engine components: Used for graph construction and GPU execution

## Current Limitations

- Only supports float32 dtype
- Limited to `arange` and `add` operations
- No testing framework currently implemented