// ══════════════════════════════════════════════════════════════════
// Intel Image Classification - Frontend JavaScript
// ══════════════════════════════════════════════════════════════════

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImage');
const predictBtn = document.getElementById('predictBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsCard = document.getElementById('resultsCard');
const resultClass = document.getElementById('resultClass');
const resultConfidence = document.getElementById('resultConfidence');
const modelUsed = document.getElementById('modelUsed');
const probabilities = document.getElementById('probabilities');

let selectedFile = null;

// ══════════════════════════════════════════════════════════════════
// File Upload Handlers
// ══════════════════════════════════════════════════════════════════

// Click to upload
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// File selected
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        alert('Please drop an image file (PNG, JPG, JPEG, or GIF)');
    }
});

// Remove image
removeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// ══════════════════════════════════════════════════════════════════
// File Handling
// ══════════════════════════════════════════════════════════════════

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        alert('Invalid file type. Please upload PNG, JPG, JPEG, or GIF');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        alert('File too large. Maximum size is 16MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'block';
        predictBtn.disabled = false;
        resultsCard.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    imageInput.value = '';
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    predictBtn.disabled = true;
    resultsCard.style.display = 'none';
}

// ══════════════════════════════════════════════════════════════════
// Prediction
// ══════════════════════════════════════════════════════════════════

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Get selected model
    const modelType = document.querySelector('input[name="model"]:checked').value;

    // Show loading
    loadingOverlay.classList.add('active');

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model', modelType);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Prediction error:', error);
    } finally {
        loadingOverlay.classList.remove('active');
    }
});

// ══════════════════════════════════════════════════════════════════
// Display Results
// ══════════════════════════════════════════════════════════════════

function displayResults(result) {
    // Show results card
    resultsCard.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

    // Display main result
    resultClass.textContent = result.class;
    resultConfidence.textContent = result.confidence.toFixed(2) + '%';
    modelUsed.textContent = result.model;

    // Color code confidence
    const confidenceColor = getConfidenceColor(result.confidence);
    resultConfidence.style.color = confidenceColor;

    // Display all probabilities
    displayProbabilities(result.all_probabilities);
}

function displayProbabilities(probs) {
    // Sort by probability
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

    // Clear previous
    probabilities.innerHTML = '';

    // Create probability bars
    sorted.forEach(([className, prob]) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';

        probItem.innerHTML = `
            <div class="prob-label">${className}</div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${prob}%">
                    <span class="prob-value">${prob.toFixed(1)}%</span>
                </div>
            </div>
        `;

        probabilities.appendChild(probItem);
    });

    // Animate bars
    setTimeout(() => {
        document.querySelectorAll('.prob-bar').forEach(bar => {
            bar.style.width = bar.style.width;
        });
    }, 100);
}

function getConfidenceColor(confidence) {
    if (confidence >= 90) return '#10b981'; // green
    if (confidence >= 70) return '#f59e0b'; // yellow
    return '#ef4444'; // red
}

// ══════════════════════════════════════════════════════════════════
// Initialize
// ══════════════════════════════════════════════════════════════════

console.log('Intel Image Classification - Ready');
