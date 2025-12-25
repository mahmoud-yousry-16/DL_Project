// Get DOM elements
const uploadSection = document.getElementById('uploadSection');
const imageInput = document.getElementById('imageInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorDiv = document.getElementById('error');
const apiUrlInput = document.getElementById('apiUrl');

let selectedImage = null;

// Upload section click handler
uploadSection.addEventListener('click', () => {
    imageInput.click();
});

// Drag and drop handlers
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImage(file);
    }
});

// File input change handler
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImage(file);
    }
});

// Handle image file
function handleImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        selectedImage = e.target.result;
        imagePreview.src = selectedImage;
        previewSection.classList.add('active');
        results.classList.remove('active');
        errorDiv.classList.remove('active');
    };
    reader.readAsDataURL(file);
}

// Analyze button click handler
analyzeBtn.addEventListener('click', async () => {
    const apiUrl = apiUrlInput.value.trim();
    
    // Validate API URL
    if (!apiUrl) {
        showError('من فضلك أدخل API URL من Colab!');
        return;
    }

    // Validate image selection
    if (!selectedImage) {
        showError('من فضلك اختر صورة أولاً!');
        return;
    }

    // Show loading state
    loading.classList.add('active');
    results.classList.remove('active');
    errorDiv.classList.remove('active');
    analyzeBtn.disabled = true;

    try {
        // Send request to API
        const response = await fetch(`${apiUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: selectedImage
            })
        });

        const data = await response.json();

        // Handle response
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'حدث خطأ في التحليل');
        }
    } catch (error) {
        showError('فشل الاتصال بالـ API. تأكد من:\n1. API URL صحيح\n2. Flask شغال على Colab\n3. ngrok شغال');
        console.error('Error:', error);
    } finally {
        // Hide loading state
        loading.classList.remove('active');
        analyzeBtn.disabled = false;
    }
});

// Display results function
function displayResults(data) {
    const predictionLabel = document.getElementById('predictionLabel');
    const confidence = document.getElementById('confidence');
    const activePercent = document.getElementById('activePercent');
    const fatiguePercent = document.getElementById('fatiguePercent');
    const activeFill = document.getElementById('activeFill');
    const fatigueFill = document.getElementById('fatigueFill');

    // Set prediction text
    predictionLabel.textContent = data.prediction === 'Active' ? '✅ نشيط' : '😴 مرهق';
    predictionLabel.className = 'prediction-label ' + (data.prediction === 'Active' ? 'active' : 'fatigue');
    
    // Set confidence
    confidence.textContent = `الثقة: ${data.confidence.toFixed(2)}%`;

    // Get probabilities
    const activeProb = data.probabilities.Active;
    const fatigueProb = data.probabilities.Fatigue;

    // Set percentages
    activePercent.textContent = `${activeProb.toFixed(1)}%`;
    fatiguePercent.textContent = `${fatigueProb.toFixed(1)}%`;

    // Animate progress bars
    setTimeout(() => {
        activeFill.style.width = `${activeProb}%`;
        fatigueFill.style.width = `${fatigueProb}%`;
    }, 100);

    // Show results
    results.classList.add('active');
}

// Show error message function
function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.add('active');
}