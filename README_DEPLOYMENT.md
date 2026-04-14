# Intel Image Classification Web App

A modern web application for classifying scenic images using Deep Learning models (PyTorch and TensorFlow).

## Features

- **Two Model Options**: Choose between PyTorch or TensorFlow CNN models
- **6 Image Classes**: buildings, forest, glacier, mountain, sea, street
- **Modern UI**: Beautiful, responsive design with smooth animations
- **Drag & Drop**: Easy image upload with drag-and-drop support
- **Real-time Results**: Instant classification with confidence scores
- **Probability Visualization**: View all class probabilities with animated bars

## Project Structure

```
intel-image-classifier/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── models/                 # Model files
│   ├── pytorch_model.pth
│   └── tensorflow_model.keras
├── static/
│   ├── css/
│   │   └── style.css      # Styles
│   ├── js/
│   │   └── script.js      # Frontend logic
│   └── uploads/           # Uploaded images
└── templates/
    └── index.html         # Main page
```

## Local Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd intel-image-classifier
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Model Files

Place your trained models in the `models/` directory:
- `models/conde_ansoumane_model.pth` (PyTorch model state dict)
- `models/conde_ansoumane_model.keras` (TensorFlow saved model)

### 5. Run Application

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## Deployment on PythonAnywhere

### Step 1: Create Account
1. Go to [PythonAnywhere.com](https://www.pythonanywhere.com)
2. Sign up for a free account

### Step 2: Upload Files
1. Open a Bash console
2. Clone your repository or upload files:
```bash
git clone <your-repo-url>
# OR upload via Files tab
```

### Step 3: Create Virtual Environment
```bash
cd intel-image-classifier
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Configure Web App
1. Go to "Web" tab
2. Click "Add a new web app"
3. Select "Manual configuration" → Python 3.10
4. Set source code: `/home/yourusername/intel-image-classifier`
5. Set virtualenv: `/home/yourusername/intel-image-classifier/venv`

### Step 5: Edit WSGI File
Click on WSGI configuration file and replace content with:

```python
import sys
path = '/home/yourusername/intel-image-classifier'
if path not in sys.path:
    sys.path.append(path)

from app import app as application
```

### Step 6: Set Static Files
In Web tab, add:
- URL: `/static/`
- Directory: `/home/yourusername/intel-image-classifier/static`

### Step 7: Reload
Click "Reload" button

Your app will be live at: `https://yourusername.pythonanywhere.com`

## Deployment on Fly.io

### Step 1: Install Fly CLI
```bash
# macOS
brew install flyctl

# Linux
curl -L https://fly.io/install.sh | sh

# Windows
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

### Step 2: Login
```bash
fly auth login
```

### Step 3: Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "2", "app:app"]
```

### Step 4: Create fly.toml

```toml
app = "intel-image-classifier"
primary_region = "iad"

[build]

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 1024
```

### Step 5: Deploy
```bash
fly launch
fly deploy
```

Your app will be live at: `https://intel-image-classifier.fly.dev`

## API Endpoints

### GET /
Main page with upload interface

### POST /predict
Classify an uploaded image

**Request:**
- `file`: Image file (multipart/form-data)
- `model`: "pytorch" or "tensorflow"

**Response:**
```json
{
  "class": "forest",
  "confidence": 95.32,
  "model": "pytorch",
  "image_path": "uploads/image.jpg",
  "all_probabilities": {
    "buildings": 1.2,
    "forest": 95.32,
    "glacier": 0.5,
    "mountain": 2.1,
    "sea": 0.7,
    "street": 0.18
  }
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "pytorch_loaded": true,
  "tensorflow_loaded": true,
  "classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"]
}
```

## Model Training

Models were trained on the Intel Image Classification dataset:
- **Dataset**: 6 classes (buildings, forest, glacier, mountain, sea, street)
- **Image Size**: 64x64 pixels
- **Architecture**: Simple CNN from scratch (no transfer learning)
- **Frameworks**: PyTorch & TensorFlow/Keras

## Technologies

- **Backend**: Flask, Python
- **Deep Learning**: PyTorch, TensorFlow
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: PythonAnywhere, Fly.io

## License

MIT License

## Author

Your Name

## Acknowledgments

- Intel Image Classification Dataset
- PyTorch & TensorFlow teams
- Flask framework
