# BotaniX
AI Plant Disease Detection Using EfficientNetB3

# 🌿 DeepVision · BotaniX – AI Plant Pathologist

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DeepVision · BotaniX** is an end-to-end plant disease detection system combining a high-accuracy EfficientNetB3 vision model with conversational AI (Gemini 3 / Gemma2). Built for researchers, farmers, and developers – it identifies 38 plant diseases from leaf images and provides treatment recommendations through a natural language interface.

![DeepVision Demo](assets/chart_model_comparison.png)

---

## 👥 Creators

| Name | Instagram | GitHub |
|------|-----------|--------|
| **Pratik Pokhrel** | [@pratikpokhrl](https://instagram.com/pratikpokhrl) | [pratikpokhrl](https://github.com/pratikpokhrl) |
| **Raunak Bhetwal** | [@raunakbhetwal](https://instagram.com/raunakbhetwal) | [raunakbhetwal](https://github.com/raunakbhetwal) |

---

## ✨ Features

- **🔍 Vision Model** – EfficientNetB3 (300x300) trained on 38 disease classes with **97.6% accuracy**
- **🧠 Dual AI Engine** – Choose between **Gemini 3 Flash** (cloud) or **Gemma2 via Ollama** (local)
- **💬 Conversational Interface** – Ask about treatments, prevention, symptoms in plain English
- **📊 Rich Visualizations** – Real-time neural network animation, confusion matrices, training curves
- **📚 Disease Database** – 38 crops/diseases with organic & chemical treatments, prevention tips
- **⚡ Fast Inference** – ~300ms per image on GPU, optimized preprocessing
- **🌐 Web Interface** – Clean, cinematic UI with dark theme

---

## 🏗️ Architecture

<img width="990" height="305" alt="image" src="https://github.com/user-attachments/assets/b92940a9-78c6-496e-b8aa-28734bbdc3d3" />

### Model Performance
| Model | Accuracy |
|-------|----------|
| VGG16 | 88.5% |
| ResNet50 | 92.4% |
| **EfficientNetB3 (Ours)** | **97.6%** |

![Confusion Matrix](assets/chart_confusion_matrix.png)
![Training Curves](assets/chart_training_curve.png)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- TensorFlow 2.15+
- (Optional) Ollama for local LLM
- (Optional) Google Gemini API key

### Installation

```bash
# Clone repository
git clone https://github.com/Spongly/BotaniX
cd BotaniX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download/place model files
# Place model.keras or best_model_phase1.keras in root directory
