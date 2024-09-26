# SolarSense: Solar Panel Monitoring System

**SolarSense** is an IoT-based system that utilizes computer vision to monitor the cleanliness of solar panels. It employs a camera attached to a Raspberry Pi to periodically capture images of the solar panels. A fine-tuned computer vision model runs on the Raspberry Pi to detect whether the panels are dirty. If the model detects dirt, it triggers a notification system to alert the user for cleaning.

## Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Raspberry Pi Setup](#raspberry-pi-setup)
  - [Deploying the Computer Vision Model](#deploying-the-computer-vision-model)
  - [API Deployment on AWS Lambda](#api-deployment-on-aws-lambda)
  - [MQTT Broker Setup](#mqtt-broker-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **SolarSense** system is designed to improve the efficiency of solar panels by detecting when they are dirty and notifying the user when cleaning is necessary. The system works as follows:

- **Raspberry Pi**: Periodically captures images of the solar panels and uses a pre-trained machine learning model to detect dirt. If dirt is detected, the system sends an MQTT message to an AWS Lambda function.
- **AWS Lambda**: The Lambda function processes the incoming MQTT message and sends a notification (e.g., email or SMS) to the user.

## Project Architecture
[Raspberry Pi + Camera] ----MQTT----> [AWS Lambda Notification Service] | v [Notification (Email/SMS/Push)]

- **Raspberry Pi**: Runs the computer vision model and publishes results via MQTT.
- **AWS Lambda**: Listens for MQTT messages, processes them, and triggers the notification service.