function toggleMenu() {
  document.querySelector('.nav-links').classList.toggle('active');
}

const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const resultBox = document.getElementById('resultBox');
const loading = document.getElementById('loading');

if (dropZone) {
  dropZone.addEventListener('click', () => imageInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#f59e0b';
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#3b82f6';
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#3b82f6';
    const files = e.dataTransfer.files;
    if (files.length) {
      imageInput.files = files;
      handleImageUpload(files[0]);
    }
  });

  imageInput.addEventListener('change', () => {
    if (imageInput.files.length) {
      handleImageUpload(imageInput.files[0]);
    }
  });
}

function handleImageUpload(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewContainer.classList.remove('hidden');
    resultBox.classList.add('hidden');
    uploadImage(file);
  };
  reader.readAsDataURL(file);
}

async function uploadImage(file) {
  loading.classList.remove('hidden');

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    loading.classList.add('hidden');

    if (data.success) {
      resultBox.classList.remove('hidden');
      document.getElementById('predictedClass').textContent = data.prediction;
      document.getElementById('confidence').textContent = data.confidence.toFixed(2) + '% confidence';

      const allModelsDiv = document.getElementById('allModels');
      allModelsDiv.innerHTML = '';

      for (const [model, result] of Object.entries(data.all_models)) {
        const div = document.createElement('div');
        div.className = 'model-prediction';
        div.innerHTML = `<strong>${model}</strong><br>${result.class} (${result.confidence.toFixed(2)}%)`;
        allModelsDiv.appendChild(div);
      }
    } else {
      alert('Error: ' + (data.error || 'Something went wrong'));
    }
  } catch (error) {
    loading.classList.add('hidden');
    alert('Error uploading image: ' + error.message);
  }
}
