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

## API Service
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

The IoT component runs on a Raspberry Pi 3, handling a scheduled image capture, model inference, and publishing messages to AWS MQTT broker when necessary. It runs a local computer vision model to classify solar panel images as either clean or dirty.

![MQTT Broker](mqtt.png)

### Key Components

1) The `CameraService` class captures images from the Raspberry Pi camera.

- **capture_image()**: Captures an image using the Raspberry Pi camera with OpenCV.
- **dummy_image()**: Loads a static image for testing purposes from the local filesystem.

2) The `ImageProcessor` class handles the preprocessing of captured images to prepare them for model inference.

- **preprocess_image()**: Resizes the image to (224, 224) and normalizes the pixel values to the range \[0, 1].

3) The `ModelService` class is responsible for loading the machine learning model and running inference on the captured image.

- **_load_model()**: Loads the pre-trained TensorFlow model from the provided model path.
- **run_inference()**: Runs inference on a preprocessed image and returns a binary prediction (0 = clean, 1 = dirty).

4) The `predict.py` script orchestrates the process of capturing an image, running inference, and sending a notification via AWS IoT if the panel is dirty. If the ML model predicts the panel is dirty (prediction == 1), the script publishes an MQTT message to the AWS IoT topic.

- **Capture Image**: Uses the `CameraService` to capture an image from the Raspberry Pi camera.
- **Run Inference**: The captured image is processed by the `ModelService` to classify it as clean (0) or dirty (1).
- **MQTT Messaging**: If the panel is classified as dirty (`prediction == 1`), the system publishes an MQTT message to AWS IoT to trigger a notification.

## 3. Computer Vision Model

The machine learning component of SolarSense handles the training, evaluation, and inference for the model which runs on the IoT device to classify solar panel images as clean or dirty.

<div align="center">
 <img src="mnv2.png" alt="MobileNetV2">
</div>

### Key Components

1) The configuration file `config-defaults.yaml` defines key parameters for model training, such as the learning rate, batch size, and input image size. These configurations are dynamically loaded during model training using the wandb integration to track experiments.

2) WandB Tracking is integrated to log and monitor the training process.

3) A MobileNetV2 pretrained on ImageNet is used as a backbone model because it is a lightweight model designed for use on edge devices.


