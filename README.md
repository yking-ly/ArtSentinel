🖼️ ArtSentinel
AI-Powered Artwork Authenticity Classifier

Detect whether a piece of art is AI-generated or human brush-made using deep learning.

🌟 Overview

ArtSentinel is a deep learning system designed to distinguish AI-generated artwork from real human-drawn art.
It leverages a fine-tuned EfficientNet-B0 CNN model trained on the AI-ArtBench dataset — containing over 180,000+ art images from both human artists and AI generators like Latent Diffusion and Stable Diffusion.

The project features:

🧠 Model training pipeline (PyTorch + Timm)

🧪 Validation, ROC-AUC, Precision-Recall metrics

⚙️ FastAPI-based inference backend

🔄 Hot model reload support

📊 Confusion matrix and results visualization

🧰 Tech Stack
Category	Tools / Frameworks
Programming Language	Python 3.10
Deep Learning	PyTorch, Timm
Data Processing	Albumentations, Pillow, NumPy, scikit-learn
Web Backend (Inference)	FastAPI, Uvicorn
Environment	.env, virtualenv
Version Control	Git + GitHub
Visualization	Matplotlib, Seaborn
Metrics	ROC-AUC, PR-AUC, Confusion Matrix
🏗️ Project Structure
ArtSentinel/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry
│   │   ├── model_timm_infer.py      # Model inference logic
│   │   └── __init__.py
│   ├── .env                         # Model + server config
│   └── requirements.txt             # Backend dependencies
│
├── training/
│   ├── train_binary_timm.py         # Main training script
│   ├── create_binary_dataset.py     # Merge AI/human folders into binary
│   ├── audit_dataset.py             # Dataset quality & duplicate audit
│   └── .venv/                       # Virtual environment
│
├── models/
│   └── runs/
│       └── effb0_full/
│           └── best.pth             # Final trained model weights
│
├── data/
│   └── binary/                      # Dataset (ignored in repo)
│       ├── train/
│       └── test/
│
├── README.md                        # This file
└── .gitignore

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/<your-username>/ArtSentinel.git
cd ArtSentinel

2️⃣ Setup Virtual Environment
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt

3️⃣ Configure Environment

Create .env inside backend/:

MODEL_NAME=efficientnet_b0
MODEL_PATH=C:\ArtSentinel\models\runs\effb0_full\best.pth
MODEL_INPUT_SIZE=224
HOST=127.0.0.1
PORT=8000

4️⃣ Run FastAPI Server
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000


API available at → http://127.0.0.1:8000/docs

🧠 Model Training

Model trained using:

Base Architecture: EfficientNet-B0 (pretrained on ImageNet)

Input Size: 224×224

Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)

Scheduler: OneCycle learning rate policy

Loss Function: CrossEntropyLoss

Precision: Mixed (float16 with AMP)

Early Stopping: Patience = 3 (on ROC-AUC)

Command Used
python train_binary_timm.py ^
  --data C:\ArtSentinel\data\binary ^
  --model_name efficientnet_b0 ^
  --pretrained ^
  --epochs 12 ^
  --batch 32 ^
  --input_size 224 ^
  --patience 3 ^
  --model_out C:\ArtSentinel\models\runs\effb0_full\best.pth ^
  --eval_test

📊 Performance Summary
Dataset	Accuracy	ROC-AUC	PR-AUC	F1-Score	Train Time
Validation	99.63%	0.9999	0.9998	0.996	~87 min
Test	99.55%	1.0000	0.9999	0.995	
Confusion Matrix (Test)
	Predicted Bot-Made	Predicted Brush-Made
Bot-Made	19,992	8
Brush-Made	127	9,873
📈 Key Insights

✅ Excellent separation of AI vs human art

⚡ Mixed-precision and OneCycle LR drastically improved training efficiency

🧩 Early stopping prevented overfitting

🧠 Validation ROC-AUC plateau detection ensured stable model checkpoints

🧍‍♂️ Model generalizes across 30 distinct artistic styles (Renaissance → Surrealism)

🔍 Example Output
API Response (JSON)
{
  "label": "Brush-Made",
  "score": 0.9876,
  "modelVersion": "effb0_full_v1",
  "processingMs": 123
}

Swagger Interface

👉 Visit http://127.0.0.1:8000/docs

🧩 Architecture Flow

Dataset Creation (create_binary_dataset.py)
→ Merges 30 folders into two classes: Bot-Made & Brush-Made

Training (train_binary_timm.py)
→ Pretrained EfficientNet fine-tuned on binary art dataset

Validation
→ ROC-AUC monitored with early stopping and auto-save of best weights

Model Saving
→ Saved checkpoint contains model name, input size, and class names

Inference (FastAPI)
→ Loads .pth weights → Preprocesses incoming image → Predicts label + confidence

📁 Example Visualization

You can include your graphs here:

test_confusion_matrix.png

metrics_comparison.png

test_error_breakdown.png

Example Markdown:

### Validation Confusion Matrix
![Validation Confusion Matrix](assets/val_confusion_matrix.png)

### Performance Comparison
![Metrics Comparison](assets/metrics_comparison.png)

🚀 API Endpoints
Endpoint	Method	Description
/health	GET	Check server health
/model-info	GET	Get current model metadata
/predict	POST	Upload image for classification
/reload	POST	Reload a new model checkpoint dynamically
💡 Future Enhancements

🧩 Integrate explainability (Grad-CAM visualization)

🕸️ Add front-end interface (React / Next.js)

🔍 Support multi-class attribution (e.g., detect which AI model generated the image)

☁️ Deploy on HuggingFace Spaces or Vercel

🧾 Citation / Credits

Dataset: AI-ArtBench

Base architecture: EfficientNet-B0 (Timm)

👨‍💻 Author

Yash Kotian
AI/ML Developer | Deep Learning Researcher
📧 yash.kotian@example.com

🌐 GitHub: YashKotian

🪪 License

MIT License © 2025 Yash Kotian