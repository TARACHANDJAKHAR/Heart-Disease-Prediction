document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const predictionResult = document.getElementById('predictionResult');
    const predictionMessage = document.getElementById('predictionMessage');
    const predictionProbability = document.getElementById('predictionProbability');
    const errorAlert = document.getElementById('errorAlert');

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
}); 