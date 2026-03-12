# 🌿 BotaniX: AI Plant Pathologist – Complete Documentation

**DeepVision · BotaniX** is an end-to-end plant disease detection system that combines deep learning (EfficientNetB3) with conversational AI. It identifies 38 plant diseases from leaf images and provides intelligent treatment recommendations through natural language interaction.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features & Capabilities](#features--capabilities)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Usage Guide](#usage-guide)
8. [API Reference](#api-reference)
9. [Model Details](#model-details)
10. [Disease Database](#disease-database)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [License](#license)

---
## 👥 Creators

| Name | Instagram | GitHub |
|------|-----------|--------|
| **Pratik Pokhrel** | [@pratikpokhrl](https://instagram.com/pratikpokhrl) | [spongly](https://github.com/Spongly) |
| **Raunak Bhetwal** | [@raunakbhetwal](https://instagram.com/raunakbhetwal) | [banwore](https://github.com/banwore) |

---

### Model Performance
| Model | Accuracy |
|-------|----------|
| VGG16 | 88.5% |
| ResNet50 | 92.4% |
| **EfficientNetB3 (Ours)** | **97.6%** |

![Confusion Matrix](assets/chart_confusion_matrix.png)
![Training Curves](assets/chart_training_curve.png)

---

## 🎯 Project Overview

### What is BotaniX?

BotaniX is an intelligent agricultural tool designed to help farmers, researchers, and gardeners identify plant diseases from leaf images and receive actionable treatment recommendations. By combining computer vision and natural language processing, it provides a user-friendly interface for disease diagnosis and management.

### Key Goals

- Democratize plant disease diagnosis using AI
- Provide fast, accurate detection (~97.6% accuracy on 38 disease classes)
- Offer contextual, conversational advice about treatments and prevention
- Support both cloud-based (Gemini) and local (Ollama) LLM backends
- Create an accessible web interface for non-technical users

### Target Users

- 👨‍🌾 Farmers & agricultural professionals
- 🔬 Plant researchers & pathologists
- 🏡 Home gardeners & plant enthusiasts
- 📱 Mobile app developers

---

## ✨ Features & Capabilities

### Core Features

#### 🔍 **Vision-Based Disease Detection**
- EfficientNetB3 CNN trained on 38 plant disease classes
- 300×300 pixel input resolution
- **97.6% accuracy** on validation set
- ~300ms inference time per image (GPU-optimized)
- Real-time preprocessing pipeline

#### 🧠 **Dual AI Engine**
- **Gemini 2.0 Flash**: Cloud-based, state-of-the-art fast responses, requires API key
- **Gemma2 via Ollama**: Local inference, privacy-preserving, offline capable
- Seamless fallback between engines
- Configurable temperature and response length

#### 💬 **Conversational Interface**
- Natural language queries about detected diseases
- Multi-turn dialogue support
- Context-aware responses
- Questions like:
  - "What are the best treatments?"
  - "How do I prevent this?"
  - "What are the symptoms?"
  - "What's the disease severity?"

#### 📊 **Rich Visualizations**
- Real-time neural network animation during inference
- Confusion matrices for model validation
- Training curves (loss, accuracy over epochs)
- Confidence scores and disease probability distributions
- Class activation maps for interpretability

#### 📚 **Disease Database**
- 38 crop-disease combinations
- For each disease:
  - Scientific and common names
  - Symptoms and characteristics
  - Organic treatments
  - Chemical treatments
  - Prevention strategies
  - Severity levels
  - Affected crop types

#### ⚡ **Performance Optimizations**
- GPU acceleration (CUDA/cuDNN support)
- Model quantization ready
- Batch processing capability
- Efficient preprocessing (ImageNet normalization)

#### 🌐 **Web Interface**
- Clean, cinematic dark-theme UI
- Responsive design (mobile & desktop)
- Drag-and-drop image upload
- Real-time model predictions
- Historical diagnosis logs
- Export capabilities

---

## 🏗️ System Architecture

### High-Level Architecture
<img width="990" height="305" alt="image" src="https://github.com/user-attachments/assets/81431ec8-acdc-4a7e-8b85-0e09682f232f" />


### Data Flow

1. **Image Input** → User uploads leaf image via web interface
2. **Preprocessing** → Resize to 300×300, normalize (ImageNet stats)
3. **Vision Model** → EfficientNetB3 inference → Disease class + confidence
4. **AI Engine** → Gemini/Ollama generates treatment recommendations
5. **UI Output** → Display disease, confidence, treatments, and Q&A interface

### Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| Frontend | User interface | HTML5, CSS3, JavaScript |
| Backend API | Request handling | Flask 2.3+ |
| Vision Model | Disease detection | TensorFlow/Keras, EfficientNetB3 |
| LLM | Contextual responses | Google Gemini API or Ollama |
| Database | Disease information | JSON/CSV or SQL |
| Storage | Model files, assets | Local filesystem |

---

## 🛠️ Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.9+ | Language runtime |
| TensorFlow | 2.15+ | Deep learning framework |
| NumPy | Latest | Numerical computations |
| Pillow (PIL) | Latest | Image processing |
| Flask | 2.3+ | Web framework |
| Requests | Latest | HTTP client |
| Ollama | (optional) | Local LLM |
| Google-Generative-AI | Latest | Gemini API client |

### Optional Dependencies

- **CUDA Toolkit** – For GPU acceleration (recommended)
- **cuDNN** – For optimized deep learning
- **Ollama** – For local Gemma2 inference

### Development Tools

- Git for version control
- Virtual environments (venv)
- pip for package management

---

## 📦 Installation & Setup

### Prerequisites

Before installation, ensure you have:
- **Python 3.9+** installed
- **pip** (Python package manager)
- **~2 GB free disk space** (for model + dependencies)
- **GPU** (optional but recommended for inference speed)

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Spongly/BotaniX.git
cd BotaniX
```

#### 2. Create Virtual Environment

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Acquire Model Files

The EfficientNetB3 model must be placed in the project root.

- Download `model.keras` from project releases or storage.
- Place in the root directory: `./model.keras`

#### 5. Configure API Keys

Create a `.env` file in the project root based on `.env.example`:

```bash
cp .env.example .env
```

#### 6. Setup Ollama (Optional, for local LLM)

```bash
# Install Ollama from https://ollama.ai
# Pull Gemma2 model
ollama pull gemma2

# Start Ollama server (in separate terminal)
ollama serve
```

#### 7. Verify Installation

```bash
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import flask; print('Flask:', flask.__version__)"
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
GEMINI_API_KEY=your_gemini_api_key
LLM_ENGINE=gemini  # or 'ollama' for local

# Ollama Configuration (if using local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your_secret_key_here

# Model Configuration
MODEL_PATH=./best_model_phase1.keras
INPUT_SIZE=300
BATCH_SIZE=32

# Inference Configuration
GPU_MEMORY_FRACTION=0.8
MAX_INFERENCE_TIME=5  # seconds
```

### Model Configuration

Edit `config.py` or environment variables to customize:

```python
# Input preprocessing
IMG_SIZE = 300  # EfficientNetB3 input size
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Disease classes (38 total)
NUM_CLASSES = 38

# LLM settings
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500
```

### Disease Database

The disease database is stored in `data/diseases.json` with structure:

```json
{
  "diseases": [
    {
      "id": 1,
      "name": "Early Blight",
      "crop": "Tomato",
      "scientific_name": "Alternaria solani",
      "symptoms": ["Brown spots on leaves", "Concentric rings"],
      "organic_treatment": ["Remove infected leaves", "Neem oil spray"],
      "chemical_treatment": ["Chlorothalonil", "Mancozeb"],
      "prevention": ["Crop rotation", "Proper spacing"],
      "severity": "moderate"
    }
  ]
}
```

---

## 🚀 Usage Guide

### Running the Application

#### Start the Web Server

```bash
python server.py
```

Expected output:
```
🚀 Starting Local Server...
✅ Vision Model Loaded!
✅ Disease database loaded!
 * Running on http://localhost:5000
```

Access the web interface at `http://localhost:5000`

#### Command-Line Usage (if available)

```bash
python predict.py --image leaf.jpg
```

Output:
```
Disease: Early Blight
Confidence: 98.2%
Class ID: 5
```

### Web Interface Walkthrough

#### 1. **Image Upload**
- Click "Upload Image" or drag-and-drop a leaf image
- Supported formats: JPG, PNG, WebP (max 5 MB)
- Real-time preview

#### 2. **Disease Detection**
- System analyzes image automatically
- Shows detected disease with confidence percentage
- Displays visual confidence bar
- Shows processing time

#### 3. **Treatment Recommendations**
- View organic and chemical treatment options
- Prevention strategies
- Severity assessment
- Symptoms checklist

#### 4. **Conversational Q&A**
- Ask follow-up questions in natural language:
  - "How long does treatment take?"
  - "What's the cost?"
  - "Are there resistant varieties?"
- System maintains conversation context

#### 5. **History & Export**
- View past diagnoses
- Export results as PDF or JSON
- Download treatment plans

### Example Workflow

```
1. Upload tomato leaf image
   ↓
2. System detects: "Early Blight (98.2% confidence)"
   ↓
3. View recommended treatments:
   - Organic: Neem oil, remove infected leaves
   - Chemical: Chlorothalonil spray
   ↓
4. Ask: "How often should I spray?"
   ↓
5. AI responds with detailed treatment schedule
   ↓
6. Export as PDF for reference
```

---

## 🔌 API Reference

### REST Endpoints

#### POST `/predict`
Detect disease from image upload.

**Request:**
```bash
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "disease": "Early Blight",
  "disease_id": 5,
  "confidence": 0.982,
  "class_name": "Tomato_Early_blight",
  "inference_time_ms": 287,
  "treatments": {
    "organic": ["Remove infected leaves", "Neem oil spray"],
    "chemical": ["Chlorothalonil", "Mancozeb"]
  }
}
```

#### POST `/chat`
Get conversational response about detected disease.

**Request:**
```json
{
  "question": "How long does treatment take?",
  "disease_id": 5,
  "llm_engine": "gemini"
}
```

**Response:**
```json
{
  "answer": "Treatment typically takes 7-14 days...",
  "sources": ["disease_database"],
  "confidence": 0.95
}
```

#### GET `/diseases`
List all disease classes.

**Response:**
```json
{
  "total": 38,
  "diseases": [
    {"id": 0, "name": "Healthy", "crop": "Apple"},
    {"id": 1, "name": "Apple_scab", "crop": "Apple"},
    ...
  ]
}
```

#### GET `/health`
System health check.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "llm_available": true,
  "gpu_available": true
}
```

---

## 🧠 Model Details

### EfficientNetB3 Architecture

**Why EfficientNetB3?**
- Optimal balance between accuracy (97.6%) and speed (~300ms)
- Efficient scaling (compound scaling coefficient)
- Pre-trained on ImageNet, fine-tuned on plant diseases
- Lower memory footprint than larger models

### Training Details

| Aspect | Details |
|--------|---------|
| **Dataset** | Plant Village Dataset (38 classes) |
| **Input Size** | 300×300 pixels |
| **Pre-training** | ImageNet weights |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Categorical Cross-Entropy |
| **Epochs** | ~50 |
| **Validation Split** | 20% |
| **Augmentation** | Rotation, zoom, flip, brightness |

### Model Performance Comparison

| Model | Accuracy | Inference (GPU) | Parameters |
|-------|----------|-----------------|-----------|
| VGG16 | 88.5% | 450ms | 138M |
| ResNet50 | 92.4% | 380ms | 25.5M |
| **EfficientNetB3** | **97.6%** | **300ms** | **10.8M** |

### How Inference Works

1. **Image Preprocessing**
   - Resize to 300×300
   - Normalize using ImageNet statistics
   - Convert to tensor

2. **Model Forward Pass**
   - Input through 10 main blocks
   - Feature extraction and pooling
   - Dense layers for classification

3. **Post-Processing**
   - Softmax to get probabilities
   - Select top-3 predictions
   - Apply confidence threshold

4. **Output**
   - Primary disease class
   - Confidence score
   - Alternative predictions

---

## 📚 Disease Database

### Supported Crops & Diseases (38 total)

#### Apple (4 diseases)
1. Apple_scab
2. Apple_Black_rot
3. Cedar_apple_rust
4. Healthy

#### Tomato (10 diseases)
5. Early_blight
6. Late_blight
7. Septoria_leaf_spot
8. Spider_mites
9. Target_spot
10. Yellow_leaf_curl_virus
11. Mosaic_virus
12. Bacterial_spot
13. Powdery_mildew
14. Healthy

#### Corn (4 diseases)
15. Cercospora_leaf_spot
16. Common_rust
17. Northern_leaf_blight
18. Healthy

...and more crops (Grape, Potato, Strawberry, etc.)

### Disease Information Structure

Each disease includes:

```json
{
  "disease_id": 5,
  "common_name": "Early Blight",
  "scientific_name": "Alternaria solani",
  "crop": "Tomato",
  "severity": "high",
  
  "symptoms": {
    "early_signs": "Small brown spots with concentric rings",
    "progression": "Spots enlarge, leaves yellow and drop",
    "conditions": "High humidity, moderate temperature"
  },
  
  "treatments": {
    "organic": [
      "Remove infected leaves",
      "Neem oil spray weekly",
      "Sulfur dust application"
    ],
    "chemical": [
      "Chlorothalonil (Bravo)",
      "Mancozeb + Copper",
      "Azoxystrobin"
    ]
  },
  
  "prevention": [
    "Crop rotation (2+ years)",
    "Proper plant spacing",
    "Avoid overhead watering",
    "Remove crop debris",
    "Use resistant varieties"
  ],
  
  "references": ["USDA", "Plant Village"]
}
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Issue: Model Not Found
```
Error: FileNotFoundError - best_model_phase1.keras not found
```

**Solution:**
```bash
# Download model or train new one
python train_model.py --dataset ./data
# Or manually download and place in root directory
```

#### Issue: Out of Memory (OOM)
```
Error: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # was 32

# Or limit GPU memory
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [
    tf.config.LogicalDeviceConfiguration(memory_limit=2048)
])
```

#### Issue: Slow Inference
```
Inference taking >1 second
```

**Solution:**
```bash
# Ensure GPU is being used
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU, install GPU drivers
# Or use CPU optimization (disable GPU)
export CUDA_VISIBLE_DEVICES=""
```

#### Issue: Gemini API Connection Error
```
Error: Failed to connect to Gemini API
```

**Solution:**
```bash
# Check API key
echo $GEMINI_API_KEY

# Test connection
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY')"

# Fallback to Ollama
export LLM_ENGINE=ollama
```

#### Issue: Ollama Not Running
```
Error: ConnectionError - Cannot connect to Ollama server
```

**Solution:**
```bash
# Start Ollama in separate terminal
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags

# Pull model if needed
ollama pull gemma2
```

#### Issue: Low Confidence Predictions
```
All predictions have <70% confidence
```

**Possible Causes & Solutions:**
- **Image Quality**: Ensure clear, well-lit leaf photos
- **Model Training**: Model may need retraining on your specific crop varieties
- **Unknown Disease**: Disease might not be in the 38-class model
- **Solution**: Feed user feedback loop to improve model

---

## 🤝 Contributing

### How to Contribute

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** and test thoroughly
4. **Commit with messages**: `git commit -m "Add feature X"`
5. **Push to branch**: `git push origin feature/your-feature`
6. **Create Pull Request** with detailed description

### Development Workflow

```bash
# Setup development environment
git clone https://github.com/Spongly/BotaniX.git
cd BotaniX
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Make changes
# Run tests
pytest tests/

# Format code
black .
flake8 .

# Commit and push
```

### Areas for Contribution

- 🌾 Additional crop-disease combinations
- 🔬 Model improvements and optimization
- 🌐 Frontend enhancements
- 📱 Mobile app development
- 📚 Documentation and tutorials
- 🐛 Bug fixes and optimization
- 🌍 Localization (multi-language support)

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for functions
- Add tests for new features

---

## 📄 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify the code
- ✅ Distribute
- ✅ Use privately

With conditions:
- ℹ️ Include license and copyright notice
- ℹ️ State significant changes

---

## 📞 Support & Contact

**Questions or Issues?**

- 📧 Create an issue on [GitHub Issues](https://github.com/Spongly/BotaniX/issues)
- 💬 Start a discussion on GitHub Discussions
- 📱 Contact: [Pratik Pokhrel](https://github.com/Spongly)

**Stay Updated:**
- ⭐ Star the repository
- 👁️ Watch for updates
- 🔔 Enable notifications

---

## 🎓 Learning Resources

### Related Documentation
- [TensorFlow/Keras Guide](https://tensorflow.org)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Flask Documentation](https://flask.palletsprojects.com)
- [Google Generative AI](https://ai.google.dev)
- [Plant Village Dataset](https://plantvillage.psu.edu)

### Tutorials
- How to fine-tune EfficientNetB3
- Deploying on cloud platforms (AWS, GCP)
- Creating custom disease datasets
- Integrating with mobile apps

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 97.6% |
| **Supported Diseases** | 38 classes |
| **Inference Speed** | ~300ms (GPU) |
| **Model Size** | ~40 MB |
| **Supported Crops** | 10+ varieties |
| **Languages** | Python 3.9+ |

---

## 🔄 Version History

### Current Version: 1.0.0
- ✅ EfficientNetB3 vision model (97.6% accuracy)
- ✅ Gemini 3 Flash integration
- ✅ Ollama/Gemma2 local inference
- ✅ Web interface with Flask
- ✅ 38 disease classes
- ✅ Comprehensive disease database

### Future Roadmap
- 📱 Mobile app (iOS/Android)
- 🌐 Multi-language support
- 📈 Real-time crop monitoring
- 🤖 Improved conversational AI
- ☁️ Cloud deployment templates
- 📊 Farmer analytics dashboard

---

## 🙏 Acknowledgments

- **Plant Village Dataset** for disease imagery
- **Google Research** for EfficientNet architecture
- **TensorFlow Team** for tools and frameworks
- **Open Source Community** for inspiration

---

**Last Updated:** March 2026
**Maintained by:** Pratik Pokhrel (@Spongly)

**Happy Plant Pathology! 🌿🔬**
