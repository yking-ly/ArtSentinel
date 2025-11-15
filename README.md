# 🎨 ArtSentinel — AI-Powered Artwork Authenticity Classifier

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Detect whether art is AI-generated or human brush-made using deep learning.**

Artificially generated artwork has become increasingly realistic due to advanced generative models like Stable Diffusion, Midjourney, and DALL·E. This makes it difficult for viewers, artists, and digital platforms to distinguish between human-created artwork and AI-produced images.

---

## 🌟 Introduction

**ArtSentinel** is a deep-learning–based classifier built using a fine-tuned **EfficientNet-B0** convolutional neural network. The model is trained on **180,000+ images** from the **AI-ArtBench dataset**, containing a diverse mix of both AI-generated images and real brush-made artwork.

ArtSentinel serves as a practical tool to:
- ✅ Verify content authenticity
- ✅ Protect artistic integrity
- ✅ Detect synthetic art in online platforms
- ✅ Provide transparency in digital art marketplaces

---

## 🎯 Objectives

- ✔️ Build a reliable classifier to distinguish between AI-generated and human-made artwork
- ✔️ Fine-tune a state-of-the-art deep learning model for high accuracy
- ✔️ Provide reproducible evaluation using ROC-AUC, PR-AUC, and confusion matrices
- ✔️ Develop a FastAPI backend to perform real-time image inference
- ✔️ Create a modern Next.js frontend for image upload and prediction visualization
- ✔️ Deploy as a full-stack system accessible to end-users

---

## 🧠 Solution Overview

ArtSentinel follows a complete ML pipeline:

### 1️⃣ Dataset Preparation
- AI-generated and brush-made classes from **AI-ArtBench** are converted into a balanced binary dataset
- 180,000+ labeled images split into training, validation, and test sets

### 2️⃣ Preprocessing & Augmentation
Image pipeline includes:
- Resizing to 224×224
- Normalization (ImageNet statistics)
- Random crops, flips, and rotations
- Affine transforms and color jitter
- Implemented using **Albumentations** library

### 3️⃣ Model Training
- **Architecture:** EfficientNet-B0 (pretrained on ImageNet)
- **Optimizer:** AdamW with weight decay
- **Scheduler:** OneCycleLR for optimal learning rate
- **Mixed Precision:** FP16 for faster training
- **Early Stopping:** ROC-AUC–based to prevent overfitting
- **Result:** Achieved extremely high accuracy and generalization

### 4️⃣ Backend (FastAPI)
- Loads the trained model
- Preprocesses incoming images
- Performs real-time inference
- Returns predictions with confidence scores
- CORS-enabled for frontend integration

### 5️⃣ Frontend (Next.js)
- Elegant, responsive UI for uploading artwork
- Displays classification results with confidence bars
- Real-time visual feedback and animations
- Communicates seamlessly with backend API

---

## 🛠 Tech Stack

### Machine Learning
- ![Python](https://img.shields.io/badge/-Python_3.10-3776AB?logo=python&logoColor=white) Python 3.10
- ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white) PyTorch
- **Timm** - EfficientNet-B0 implementation
- **NumPy** - Numerical computing
- **Pillow** - Image processing
- **Albumentations** - Advanced image augmentation
- **scikit-learn** - Metrics (ROC-AUC, PR-AUC, confusion matrix)

### Backend (Inference API)
- ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white) FastAPI
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **CORS Middleware** - Cross-origin support

### Frontend
- ![Next.js](https://img.shields.io/badge/-Next.js_14-000000?logo=next.js&logoColor=white) Next.js 14 (App Router)
- ![TailwindCSS](https://img.shields.io/badge/-TailwindCSS-38B2AC?logo=tailwind-css&logoColor=white) TailwindCSS
- ![React](https://img.shields.io/badge/-React-61DAFB?logo=react&logoColor=black) React Hooks
- Image dropzone with drag-and-drop
- Smooth UI animations

### Deployment
- **Vercel** - Frontend hosting
- **Railway/Render** - FastAPI backend deployment (Not yet deployed)

---

## 📊 Model Performance

The trained EfficientNet-B0 model achieves:

| Metric | Score |
|--------|-------|
| **Accuracy** | 95%+ |
| **ROC-AUC** | 0.98+ |
| **PR-AUC** | 0.97+ |
| **F1-Score** | 0.94+ |

The model demonstrates strong generalization with minimal overfitting through:
- Early stopping based on validation ROC-AUC
- Extensive data augmentation
- Transfer learning from ImageNet weights

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA-enabled GPU (optional, for training)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/artsentinel.git
cd artsentinel/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The application will be available at `http://localhost:3000`

---

## 📸 Screenshots

### Homepage Interface
![Homepage](https://github.com/user-attachments/assets/aafec1ac-fced-49b5-9675-4fa385f12fde)
<p align="center"><em>ArtSentinel Homepage - Upload and analyze artwork authenticity</em></p>

### Real Artwork Detection - Vagabond Manga
![Vagabond Classification](https://github.com/user-attachments/assets/a8bcd022-5548-40e5-82b9-e1d241b3b955)
<p align="center"><em>Vagabond manga panel correctly classified as Real Art (Human-made)</em></p>

### AI-Generated Art Detection
![Gemini AI Art](https://github.com/user-attachments/assets/82726239-2755-4763-81f4-805cb5961bd4)
<p align="center"><em>Gemini-generated artwork correctly classified as AI Art</em></p>

### Real Artwork Detection - Traditional Art
![Random Art Classification](https://github.com/user-attachments/assets/357e6f0c-1474-4195-a4ce-70e75597ed3e)
<p align="center"><em>Traditional artwork correctly classified as Real Art</em></p>

### Model Limitation - False Positive
![Ghibli Art Misclassification](https://github.com/user-attachments/assets/475fa55e-fff2-4ef1-8bc9-65fc2a0f5860)
<p align="center"><em>Studio Ghibli artwork misclassified as AI Art (False Positive) - Demonstrates model limitations with highly stylized animation</em></p>

---

## 🎯 API Endpoints

### POST `/predict`
Upload an image for classification

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@artwork.jpg"
```

**Response:**
```json
{
  "prediction": "Real Art",
  "confidence": 0.92,
  "probabilities": {
    "ai_generated": 0.08,
    "human_made": 0.92
  }
}
```

---

## 🔬 Training the Model

To train the model from scratch:

```bash
cd training

# Prepare dataset
python prepare_dataset.py --data-path /path/to/ai-artbench

# Train model
python train.py \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --model efficientnet_b0

# Evaluate model
python evaluate.py --checkpoint best_model.pth
```

---

## 📈 Future Improvements

- [ ] Add support for more art styles and mediums
- [ ] Implement attention visualization (Grad-CAM)
- [ ] Expand dataset with recent AI-generated art samples
- [ ] Add multi-class classification (Stable Diffusion, Midjourney, DALL·E, etc.)
- [ ] Integrate explainability features for predictions
- [ ] Mobile application development
- [ ] Batch processing API endpoint

---

## ⚠️ Known Limitations

- **False Positives:** Highly stylized or digital artwork (like Studio Ghibli animation) may be misclassified as AI-generated
- **Dataset Bias:** Model performance depends on the diversity of training data
- **Adversarial Attacks:** Sophisticated post-processing may fool the classifier
- **Evolving AI Art:** New generative models may produce images the model hasn't seen during training

---

## 🙏 Acknowledgments

- **AI-ArtBench Dataset** - For providing the training data
- **EfficientNet** - For the efficient model architecture
- **FastAPI** - For the robust backend framework
- **Next.js** - For the modern frontend framework
- **Open Source Community** - For the amazing tools and libraries

---

## 📧 Contact

For questions, feedback, or collaboration opportunities:

- **GitHub:** [@yking-ly](https://github.com/yking-ly)
- **Email:** yashnkotian3006@example.com
- **LinkedIn:** [Yash Kotian](https://linkedin.com/in/yashkotian/)

---

<p align="center">
  <strong>⭐ If you find this project useful, please consider giving it a star! ⭐</strong>
</p>
