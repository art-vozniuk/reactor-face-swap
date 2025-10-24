# Face Swap Module

This repository is forked from [Gourieff/ComfyUI-ReActor](https://github.com/Gourieff/ComfyUI-ReActor) to use its core face swap pipeline in my [demo](https://github.com/art-vozniuk/demo-hub) project.

## Changes from Original

This fork has been stripped down and modified to work as a standalone module:

- **Removed unused code**: All ComfyUI-related code and dependencies have been removed since I only need the core face swap functionality.

- **Dependency management**: The dependencies have been restructured to work as a uv package dependency, making it easier to integrate into the main project.

- **GPU acceleration**: The ONNX models have been configured to use GPU during inference for better performance.

## Purpose

This module serves as a helper component for the main demo application, providing face swap capabilities without the overhead of the full ComfyUI framework.

