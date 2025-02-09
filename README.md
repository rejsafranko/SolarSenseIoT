# SolarSense: Solar Panel Dirt Notification

SolarSense is an IoT and machine learning (ML) solution designed to detect and notify users when solar panels require cleaning. By leveraging a Raspberry Pi camera, an onboard TensorFlow Lite model, and AWS cloud services, SolarSense provides an automated system for maintaining optimal solar panel efficiency.

![Architecture](architecture.png)

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [API Service](#api-service)
4. [IoT Client](#iot-client)
5. [Model Training & Deployment](#model-training--deployment)
6. [Deployment](#deployment)
7. [Future Enhancements](#future-enhancements)
8. [Contributing](#contributing)
9. [License](#license)

## Overview
Solar panels accumulate dirt over time, reducing efficiency. SolarSense automates the detection process by:
- Capturing images using a Raspberry Pi camera
- Running a TensorFlow Lite ML model locally to classify images as clean or dirty
- Automatically deploying updated models via AWS S3 and MQTT
- Sending alerts via AWS SNS when cleaning is required

## System Architecture
The system consists of three key components:
1. **IoT Client:** Runs on a Raspberry Pi, captures images, and performs inference.
2. **API Service:** Handles notifications and integrates with AWS IoT and SNS.
3. **Model Training & Deployment:** Uses AWS S3 and MQTT to update IoT devices automatically.

## 1. API Service
The API is implemented as an AWS Lambda function triggered by AWS IoT. It sends notifications via AWS SNS when a dirty solar panel is detected.

### Workflow:
1. The IoT device publishes an MQTT message to AWS IoT Core.
2. AWS IoT triggers the Lambda function.
3. The Lambda function sends a notification via AWS SNS.

```typescript
import { APIGatewayProxyEvent, APIGatewayProxyResult } from "aws-lambda";
import { SNS } from "aws-sdk";
import dotenv from "dotenv";

dotenv.config();
const sns = new SNS();

export const handler = async (
  event: APIGatewayProxyEvent
): Promise<APIGatewayProxyResult> => {
  // Function implementation...
};
```

## 2. IoT Client

![MQTT Broker](mqtt.png)

The IoT client runs on a **Raspberry Pi** and is responsible for:
- Capturing images using the Raspberry Pi camera.
- Running inference using **TensorFlow Lite**.
- Publishing results to AWS IoT Core via MQTT.

### Key Components:
- **CameraService**: Captures images using the Raspberry Pi camera.
- **ImageProcessor**: Preprocesses images for inference.
- **ModelService**: Loads and runs the TensorFlow Lite model for classification.
- **MQTTClient**: Handles MQTT connectivity and message publishing.
- **predict.py**: Orchestrates image capture, inference, and MQTT messaging.

### Workflow:
1. Capture image using OpenCV.
2. Preprocess the image (resize, normalize, etc.).
3. Run inference using TensorFlow Lite.
4. If classified as **dirty**, publish an MQTT message to AWS IoT Core.

## 3. Computer Vision Model

The computer vision model is based on **MobileNetV2** and trained using **TensorFlow & Keras**.  
It is designed to classify images of solar panels as **clean** or **dirty**.

<div align="center">
 <img src="mnv2.png" alt="MobileNetV2">
</div>

### Model Architecture
The model leverages **transfer learning**, using MobileNetV2 as the feature extractor:
- **Base Model**: MobileNetV2 (pretrained on ImageNet, frozen weights).
- **Global Average Pooling Layer**: Reduces feature map size.
- **Dense Layer (1024 neurons, ReLU activation)**: Learns solar panel-specific patterns.
- **Dropout Layer**: Prevents overfitting (value set via configuration).
- **Output Layer (1 neuron, Sigmoid activation)**: Performs binary classification.

### Training Pipeline
1. **Dataset Preparation**: Images are preprocessed and split into training, validation, and test sets.
2. **Model Training**: The model is trained using an **Adam optimizer** with a **binary cross-entropy loss function**.
3. **Performance Monitoring**: Training metrics (accuracy, loss) are logged using **Weights & Biases (WandB)**.
4. **Early Stopping & Checkpointing**: Ensures the best-performing model is saved.
5. **Evaluation**: The model is tested on unseen data to measure accuracy, precision, recall, and F1-score.

### Model Deployment
- The trained model is **converted to TensorFlow Lite (TFLite)** for efficient edge deployment.
- The **optimized model is uploaded to AWS S3** for centralized distribution if it is the champion model on the test set.
- IoT devices **receive automatic updates** via **MQTT messaging** when a new model is available.



