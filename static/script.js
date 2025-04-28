document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const predictionResult = document.getElementById('predictionResult');
    const predictionMessage = document.getElementById('predictionMessage');
    const predictionProbability = document.getElementById('predictionProbability');
    const errorAlert = document.getElementById('errorAlert');
    const interpretationResult = document.getElementById('interpretationResult');
    const showInterpretationBtn = document.getElementById('showInterpretation');
    const interpretationImage = document.getElementById('interpretationImage');
    const interpretationText = document.getElementById('interpretationText');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hide previous results and errors
        predictionResult.style.display = 'none';
        errorAlert.style.display = 'none';
        
        // Add loading state
        form.classList.add('loading');
        
        // Collect form data
        const formData = {};
        const formElements = form.elements;
        for (let element of formElements) {
            if (element.name && element.value) {
                formData[element.name] = element.value;
            }
        }

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            // Show prediction result
            predictionResult.style.display = 'block';
            predictionResult.className = data.prediction === 1 ? 'card danger' : 'card success';
            predictionMessage.textContent = data.message;
            predictionMessage.className = data.prediction === 1 ? 'card-text text-danger' : 'card-text text-success';
            predictionProbability.textContent = `Probability: ${(data.probability * 100).toFixed(2)}%`;

        } catch (error) {
            // Show error message
            errorAlert.textContent = error.message;
            errorAlert.style.display = 'block';
        } finally {
            // Remove loading state
            form.classList.remove('loading');
        }
    });

    showInterpretationBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/interpretation');
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to load interpretation');
            }

            // Display the interpretation
            interpretationImage.innerHTML = `<img src="data:image/png;base64,${data.plot_image}" class="img-fluid" alt="Feature Importance Plot">`;
            interpretationText.innerHTML = `
                <div class="user-friendly mb-4">
                    <h6 class="mb-3">Simple Explanation:</h6>
                    <pre class="user-friendly-text">${data.user_friendly_interpretation}</pre>
                </div>
                <div class="technical">
                    <h6 class="mb-3">Technical Details:</h6>
                    <pre class="technical-text">${data.technical_interpretation}</pre>
                </div>
            `;
            interpretationResult.style.display = 'block';

        } catch (error) {
            errorAlert.textContent = error.message;
            errorAlert.style.display = 'block';
        }
    });
});