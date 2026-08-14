document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictButton');
    const predictSpinner = document.getElementById('predictSpinner');
    
    const predictionResult = document.getElementById('predictionResult');
    const resultBadge = document.getElementById('resultBadge');
    const predictionMessage = document.getElementById('predictionMessage');
    const predictionProbability = document.getElementById('predictionProbability');
    
    const errorAlert = document.getElementById('errorAlert');
    
    const showInterpretationBtn = document.getElementById('showInterpretation');
    const explainSpinner = document.getElementById('explainSpinner');
    const interpretationResult = document.getElementById('interpretationResult');
    const interpretationImage = document.getElementById('interpretationImage');
    const userFriendlyText = document.getElementById('userFriendlyText');
    const technicalText = document.getElementById('technicalText');

    let currentFormData = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hide previous results
        predictionResult.classList.add('hidden');
        interpretationResult.classList.add('hidden');
        errorAlert.classList.add('hidden');
        
        // UI loading state
        predictBtn.disabled = true;
        predictSpinner.classList.remove('hidden');
        
        // Collect data
        const formData = {};
        for (let element of form.elements) {
            if (element.name && element.value) {
                formData[element.name] = element.value;
            }
        }
        currentFormData = formData;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            // Populate results
            const isDanger = data.prediction === 1;
            
            resultBadge.textContent = isDanger ? 'Attention Required' : 'Normal Range';
            resultBadge.className = `badge ${isDanger ? 'danger' : 'success'}`;
            
            predictionMessage.textContent = data.message;
            predictionMessage.className = `result-message ${isDanger ? 'danger' : 'success'}`;
            
            predictionProbability.textContent = `Confidence: ${(data.probability * 100).toFixed(1)}%`;
            
            predictionResult.classList.remove('hidden');

        } catch (error) {
            errorAlert.textContent = error.message;
            errorAlert.classList.remove('hidden');
        } finally {
            predictBtn.disabled = false;
            predictSpinner.classList.add('hidden');
        }
    });

    showInterpretationBtn.addEventListener('click', async () => {
        if (!currentFormData) return;

        // Hide previous interpretation
        interpretationResult.classList.add('hidden');
        errorAlert.classList.add('hidden');

        // UI loading state
        showInterpretationBtn.disabled = true;
        explainSpinner.classList.remove('hidden');

        try {
            const response = await fetch('/api/interpretation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentFormData),
            });
            
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to load explanation');
            }

            // Display results
            interpretationImage.innerHTML = `<img src="data:image/png;base64,${data.plot_image}" alt="Feature Importance Plot">`;
            userFriendlyText.textContent = data.user_friendly_interpretation;
            technicalText.textContent = data.technical_interpretation;
            
            interpretationResult.classList.remove('hidden');
            
            // Scroll to interpretation
            interpretationResult.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            errorAlert.textContent = error.message;
            errorAlert.classList.remove('hidden');
        } finally {
            showInterpretationBtn.disabled = false;
            explainSpinner.classList.add('hidden');
        }
    });
});