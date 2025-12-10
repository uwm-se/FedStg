# Federated Learning with Dynamic Data Distribution and Client Participation

This repository contains the complete implementation of an adaptive Federated Learning (FL) framework featuring:

- Dynamic cosine-based class rotation (temporal data drift)
- Capability- and performance-aware client participation
- Rarity- and capability-weighted aggregation
- Stagnation-aware aggregation scheduling
- CIFAR-10 experiments and logged results

The project is fully self-contained within the following files:

- accuracy.png
- client.py
- server.py
- model.py
- fl_results.json

  
---

## 📁 File Descriptions

### **server.py**
Implements the FL server logic:
- Stagnation-aware aggregation (warm-up → reduced frequency → fallback)
- Weighted aggregation using:
  \[
  W_i = N_i \times S_i \times R_i
  \]
- Tracking accuracy, loss, and drift effects
- Saving training logs to `fl_results.json`

---

### **client.py**
Contains the client implementation:
- Local training on CIFAR-10
- Cosine-based dynamic data distribution per round
- Adaptive participation probability (capability × performance × decay)
- Communication stubs compatible with Flower-style FL

---

### **model.py**
Defines the lightweight CNN used across all experiments:
- 3 convolution blocks (64 → 128 → 256)
- BatchNorm + ReLU + MaxPool
- FC (512) + Dropout
- Softmax output (10 classes)

A compact model (~1.4M parameters) suitable for FL on heterogeneous devices.

---

### **fl_results.json**
Contains saved numerical logs from a complete federated run:
- Round-by-round accuracy and loss  
- Client participation data  
- Aggregation decisions  
- Final metrics  

---

### **accuracy.png**
Plot of the accuracy curve produced during training.

## 🚀 Running the Code

## Start the server
python server.py

## Start the client client
python client.py
