# Realtime Fire Detection Using YOLO Models

## Table of Achieved Evaluation Metrics

| Dataset    | Metric                     | Value                |
| ---------- | -------------------------- | -------------------- |
| Validation | **Mean Average Precision** | **77.8%** @ 0.5(IoU) |
| Validation | **Precision**              | **79.6%**            |
| Validation | **Recall**                 | **73.4%**            |

## Introduction

This repository hosts a collection of YOLO (You Only Look Once) models trained specifically for the purpose of fire detection. With the increasing risks of wildfires and urban fires, timely and accurate detection is crucial for early response and mitigation. Leveraging the speed and efficiency of YOLO models, our project aims to provide a robust solution for identifying fire instances in real-time video feeds and images.

## Why YOLO for Fire Detection?

YOLO models are renowned for their fast detection speeds, making them ideal for real-time applications. Unlike traditional methods that might process an image in multiple stages, YOLO models predict multiple bounding boxes and class probabilities in a single forward pass. This capability allows our fire detection system to operate effectively in real-time scenarios, such as surveillance cameras in forests, urban areas, and industrial settings.

## Use Cases

- **Wildfire Monitoring:** Integration with drone and satellite imagery to detect early signs of wildfires, allowing for rapid response.
- **Urban Safety:** Monitoring urban environments and infrastructure, such as buildings and highways, for early detection of fire outbreaks.
- **Industrial Safety:** Ensuring safety in industrial and manufacturing settings by detecting fires in their incipient stage, preventing potential damage and ensuring workforce safety.
- **Home Security:** Integration with home security systems to provide real-time alerts on fire incidents, enhancing personal safety and property protection.

## Benefits

- **Speed and Efficiency:** Real-time detection capabilities allow for immediate identification and response to fire incidents.
- **Accuracy:** Trained on diverse datasets, our YOLO models are capable of detecting fires with high precision, reducing false positives.
- **Scalability:** Easily integrated into existing surveillance and monitoring systems without the need for complex hardware.
- **Cost-Effectiveness:** Provides a cost-efficient solution for fire detection compared to traditional methods, with lower operational costs and resource requirements.

## Installation

Setting up the Fire Detection project is straightforward and primarily requires PyTorch and the Ultralytics package. Follow the steps below to get started.

### Requirements

- Python 3.6 or later
- PyTorch (The installation command below will install it if not already installed)
- Ultralytics

### Pip Install Method (Recommended)

To install the necessary packages, run the following command in your Python environment:

```bash
!pip install torch ultralytics==8.0.196
```

## Getting Started

To get started with deploying the fire detection models, follow the steps below:

1. **Clone the Repository:**

```bash
git clone https://github.com/pavan98765/Fire_Detection.git
```

2.
