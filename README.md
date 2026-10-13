# JetsonNano_for_Semantic_Communications

<img width="1015" height="223" alt="image" src="https://github.com/user-attachments/assets/068f4d86-a22d-49f1-93ad-76acfd2570d8" />

本仓库提供了一个端到端部署与评估框架，用于在 NVIDIA Jetson Nano 边缘设备上部署轻量级语义通信系统。  
该实现重点验证基于 DeepJSCC 的模型在资源受限边缘环境中的工程可行性、运行时稳定性与推理延迟。

<img width="1015" height="706" alt="image" src="https://github.com/user-attachments/assets/5925fe2b-8cd3-4f05-ad41-edeabe326969" />

## 1. 概述

我们部署并评估 DeepJSCC 语义通信系统。

系统部署在两台 NVIDIA Jetson Nano 设备上，分别作为发射端（Tx）和接收端（Rx），构成真实的端到端语义通信设置。

## 2. 硬件平台

每台 Jetson Nano 设备配置如下：

- GPU：NVIDIA Maxwell 架构
- CPU：四核 ARM Cortex-A57
- 内存：4 GB LPDDR4
- 操作系统：Ubuntu（JetPack 环境）

## 3. 系统架构

### 3.1 端到端语义通信

系统由两个边缘设备组成：

#### 发射端（Tx）

- 执行语义编码
- 模拟信道噪声
- 发送受信道噪声污染的语义潜在表示

#### 接收端（Rx）

- 接收语义潜在特征
- 执行语义解码与图像重建

仅传输低维语义潜在表示。  
原始图像从不发送，从而显著降低通信开销和带宽消耗。

### 3.2 通信方式

- 传输协议：TCP Socket（点对点）
- 网络设置：无线局域网（同一本地网络）

## 4. 结论

本仓库表明，语义通信模型不仅在仿真中表现良好，还能在真实边缘设备上稳定、高效地运行。
