# 🚀 COMPLETE DEPLOYMENT GUIDE
## Intel Image Classification Web App

---

## 📦 What You Have

A **complete Flask web application** with:
- ✅ Beautiful, modern UI
- ✅ PyTorch & TensorFlow model support
- ✅ Drag & drop image upload
- ✅ Real-time classification
- ✅ Responsive design
- ✅ Production-ready code

---

## 🏗️ Project Structure

```
intel-image-classifier/
├── app.py                      # Flask backend
├── requirements.txt            # Dependencies
├── Dockerfile                  # Docker config
├── fly.toml                    # Fly.io config
├── .gitignore                  # Git ignore rules
├── create_dummy_models.py      # Test model generator
├── models/                     # Your trained models go here
│   ├── pytorch_model.pth       # PyTorch model
│   └── tensorflow_model.keras  # TensorFlow model
├── static/
│   ├── css/
│   │   └── style.css          # Beautiful CSS
│   ├── js/
│   │   └── script.js          # Frontend logic
│   └── uploads/               # Upload folder
└── templates/
    └── index.html             # Main page
```

---

## 🎯 QUICK START (Local Testing)

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Create Test Models

```bash
# Generate dummy models for testing
python create_dummy_models.py
```

### Step 3: Run App

```bash
python app.py
```

Visit: **http://127.0.0.1:5000**

---

## 📤 DEPLOYMENT OPTION 1: PythonAnywhere (FREE)

### Why PythonAnywhere?
- ✅ **100% FREE** tier available
- ✅ Easy setup (no Docker needed)
- ✅ Perfect for beginners
- ❌ Limited resources (1 web app, 512MB RAM)

### Step-by-Step:

#### 1. Create Account
- Go to https://www.pythonanywhere.com
- Sign up for FREE account
- Confirm email

#### 2. Upload Files

**Option A: Git (Recommended)**
```bash
# In PythonAnywhere Bash console:
git clone https://github.com/your-username/intel-classifier.git
cd intel-classifier
```

**Option B: Manual Upload**
- Go to "Files" tab
- Click "Upload a file"
- Upload all project files

#### 3. Create Virtual Environment

```bash
# In Bash console:
cd intel-classifier
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Upload Your Models

- Go to "Files" tab
- Navigate to `intel-classifier/models/`
- Upload your trained models:
  - `pytorch_model.pth`
  - `tensorflow_model.keras`

#### 5. Configure Web App

1. Go to **"Web"** tab
2. Click **"Add a new web app"**
3. Choose your username.pythonanywhere.com
4. Select **"Manual configuration"**
5. Choose **Python 3.10**

#### 6. Set Paths

In Web tab configuration:

**Source code:**
```
/home/YOUR_USERNAME/intel-classifier
```

**Virtualenv:**
```
/home/YOUR_USERNAME/intel-classifier/venv
```

#### 7. Configure WSGI File

Click on **WSGI configuration file** link, delete all content, paste:

```python
import sys
import os

# Add project directory
path = '/home/YOUR_USERNAME/intel-classifier'
if path not in sys.path:
    sys.path.insert(0, path)

# Set Flask app
os.chdir(path)
from app import app as application
```

Replace `YOUR_USERNAME` with your actual PythonAnywhere username!

#### 8. Configure Static Files

In Web tab, scroll to **"Static files"** section:

Add entry:
- **URL:** `/static/`
- **Directory:** `/home/YOUR_USERNAME/intel-classifier/static`

#### 9. Reload & Launch

- Click big green **"Reload"** button
- Visit: `https://YOUR_USERNAME.pythonanywhere.com`

🎉 **Your app is live!**

---

## 🚀 DEPLOYMENT OPTION 2: Fly.io (FREE)

### Why Fly.io?
- ✅ FREE tier (3 apps)
- ✅ Global CDN
- ✅ Docker-based (more control)
- ✅ Better performance
- ❌ Requires credit card (not charged on free tier)

### Step-by-Step:

#### 1. Install Fly CLI

**Mac:**
```bash
brew install flyctl
```

**Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

**Windows:**
```powershell
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

#### 2. Login

```bash
fly auth login
```

#### 3. Prepare Project

Make sure your models are in `models/` folder:
```
models/
├── pytorch_model.pth
└── tensorflow_model.keras
```

#### 4. Launch App

```bash
# In project directory:
fly launch

# Answer prompts:
# - App name: intel-classifier (or your choice)
# - Region: Choose closest to you
# - PostgreSQL: No
# - Redis: No
```

#### 5. Deploy

```bash
fly deploy
```

#### 6. Check Status

```bash
fly status
fly logs
```

Your app will be live at: `https://intel-classifier.fly.dev`

---

## 🔧 Replacing Dummy Models with Real Models

### From Your Jupyter Notebook:

#### Save PyTorch Model:
```python
# After training in notebook:
torch.save(model.state_dict(), 'pytorch_model.pth')
```

Download `pytorch_model.pth` and place in `models/` folder.

#### Save TensorFlow Model:
```python
# After training in notebook:
model.save('tensorflow_model.keras')
```

Download `tensorflow_model.keras` and place in `models/` folder.

---

## 🧪 Testing Your Deployment

### Test Locally:
```bash
python app.py
# Visit http://127.0.0.1:5000
```

### Test Health Endpoint:
```bash
curl https://your-app-url/health
```

Should return:
```json
{
  "status": "ok",
  "pytorch_loaded": true,
  "tensorflow_loaded": true,
  "classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"]
}
```

---

## 🐛 Troubleshooting

### "Model not loaded" error:
- ✅ Check models are in `models/` folder
- ✅ Check file names are exact: `pytorch_model.pth`, `tensorflow_model.keras`
- ✅ Verify models match the architecture in `app.py`

### "Memory error" on deployment:
- Reduce model size or use CPU-only versions
- On Fly.io: increase VM memory in `fly.toml`

### Static files not loading:
- **PythonAnywhere**: Check Static files configuration
- **Fly.io**: Verify Dockerfile copies static folder

### "Port already in use":
```bash
# Kill process using port 5000
# Mac/Linux:
lsof -ti:5000 | xargs kill -9
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

---

## 📊 Model Requirements

Your models MUST:
1. Accept **64×64 RGB images**
2. Output **6 classes** (buildings, forest, glacier, mountain, sea, street)
3. Use **same preprocessing** as in app.py:
   - Resize to 64×64
   - Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## 🎨 Customization

### Change Colors:
Edit `static/css/style.css`, modify CSS variables:
```css
:root {
    --primary: #6366f1;    /* Change main color */
    --secondary: #10b981;  /* Change accent */
}
```

### Add More Classes:
1. Update `CLASSES` list in `app.py`
2. Retrain models with new classes
3. Update model output layer

### Modify UI:
Edit `templates/index.html` and `static/css/style.css`

---

## 📝 Final Checklist

Before deployment:

- [ ] Models trained and saved
- [ ] Models placed in `models/` folder
- [ ] Tested locally with `python app.py`
- [ ] requirements.txt has all dependencies
- [ ] .gitignore prevents uploading large files
- [ ] Static files configured correctly
- [ ] WSGI file has correct paths (PythonAnywhere)
- [ ] App deployed and accessible

---

## 🎓 Learning Resources

**Flask:**
- https://flask.palletsprojects.com

**PythonAnywhere:**
- https://help.pythonanywhere.com

**Fly.io:**
- https://fly.io/docs

**PyTorch:**
- https://pytorch.org/docs

**TensorFlow:**
- https://www.tensorflow.org/guide

---

## 💡 Tips

1. **Start with PythonAnywhere** - easiest for beginners
2. **Test locally first** - debug before deploying
3. **Use version control** - git commit often
4. **Monitor logs** - check for errors
5. **Backup models** - keep copies of trained models

---

## 🆘 Need Help?

Common issues and solutions are in the Troubleshooting section above.

For more help:
- PythonAnywhere: forums.pythonanywhere.com
- Fly.io: community.fly.io
- Stack Overflow: stackoverflow.com

---

## 🎉 You're Ready!

Your complete Flask application is ready for deployment. Follow the guide above and you'll have a live web app in minutes!

**Good luck! 🚀**
