FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt -y install cmake

WORKDIR /ODT