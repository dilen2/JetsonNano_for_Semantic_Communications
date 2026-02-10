# Semantic-Communication-Deployment-on-Jetson-Nano

This repository provides an end-to-end deployment and evaluation framework for lightweight semantic communication systems on NVIDIA Jetson Nano edge devices.
The implementation focuses on validating the engineering feasibility, runtime stability, and inference latency of DeepJSCC-based models under resource-constrained edge environments.

1. Overview

We deploy and evaluate the following three semantic communication systems:

DeepJSCC

DeepJSCC + CPAM

DeepJSCC + CPAM-l (lightweight variant)

The systems are deployed on two NVIDIA Jetson Nano devices, acting as the transmitter (Tx) and receiver (Rx) respectively, forming a real end-to-end semantic communication setup.

Unlike offline simulations on high-performance GPUs, this repository emphasizes real-world edge deployment, where memory capacity, computing power, and energy efficiency are strictly limited.

2. Hardware Platform

Each Jetson Nano device is configured as follows:

GPU: NVIDIA Maxwell architecture

CPU: Quad-core ARM Cortex-A57

Memory: 4 GB LPDDR4

OS: Ubuntu (JetPack environment)

This hardware configuration represents a typical edge communication node, making it suitable for evaluating lightweight semantic communication models in practical scenarios.

3. System Architecture
3.1 End-to-End Semantic Communication

The system consists of two edge devices:

Transmitter (Tx)

Performs semantic encoding

Simulates channel noise

Sends corrupted semantic latent representations

Receiver (Rx)

Receives semantic latent features

Performs semantic decoding and image reconstruction

Only low-dimensional semantic latent representations are transmitted.
Raw images are never sent, significantly reducing communication overhead and bandwidth consumption.

3.2 Communication Method

Transmission protocol: TCP Socket (point-to-point)

Network setup: Wireless LAN (same local network)

Measured bandwidth:

Download: ~12.33 Mbps

Upload: ~36.75 Mbps

Experimental results show that, under the tested conditions, network transmission has minimal impact on overall system performance, and the dominant latency comes from model inference.

4. Deployment Strategy on Jetson Nano

Due to the 4 GB memory limitation, the following deployment strategies are adopted:

Inference-only execution

Gradient computation is disabled (torch.no_grad())

Frame-by-frame decoding

Avoids peak memory usage caused by batch processing

Lightweight model design

CPAM-l further reduces inference overhead while preserving performance gains

Under these constraints, all three systems run stably without memory overflow or inference interruption, demonstrating the resource-controllable nature of the CPAM modules.

5. Latency Measurement

To evaluate real-time performance, inference latency is measured as follows:

Measurement granularity: Per-frame decoding

Number of frames: 100 consecutive images

Initialization handling:

The first frame is excluded to avoid cache and loading overhead

Final metric: Average decoding latency per frame

This setup provides a fair and reproducible latency evaluation for edge semantic communication systems.

6. Key Findings

CPAM and CPAM-l can be successfully deployed on memory-constrained edge devices

CPAM-l achieves a better trade-off between performance gain and inference latency

The system demonstrates stable operation under real wireless network conditions

Results validate that the proposed models are not only effective in simulation but also practically deployable in real edge scenarios

This repository demonstrates that semantic communication models can not only work well in simulations, but can also run stably and efficiently on real edge devices.        
