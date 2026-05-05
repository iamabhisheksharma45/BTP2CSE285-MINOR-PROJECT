document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.getElementById('loading-spinner');
    const resetBtn = document.getElementById('reset-btn');
    
    let probabilityChart = null;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        predictBtn.disabled = true;
        
        // Collect data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            
            const result = await response.json();
            
            // Artificial delay for animation effect
            setTimeout(() => {
                showResult(result);
            }, 800);
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while making the prediction. Please try again.');
            resetForm();
        }
    });
    
    resetBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        form.classList.remove('hidden');
        form.reset();
        
        // Slight animation class reset
        form.style.animation = 'none';
        form.offsetHeight; /* trigger reflow */
        form.style.animation = 'fadeInUp 0.5s ease-out';
    });
    
    function showResult(data) {
        form.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        
        const titleElement = document.getElementById('result-title');
        const messageElement = document.getElementById('result-message');
        const iconElement = document.getElementById('result-icon');
        const probPercentage = document.getElementById('prob-percentage');
        
        if (data.success) {
            const prob = data.probability ? parseFloat(data.probability).toFixed(1) : (data.prediction === 1 ? 100 : 0);
            probPercentage.textContent = `${prob}%`;
            
            if (data.prediction === 1) {
                // High Risk
                titleElement.textContent = 'High Risk Detected';
                titleElement.className = 'risk-high';
                iconElement.textContent = '⚠️';
                messageElement.textContent = 'Based on the provided parameters, the model indicates a high risk of heart disease. We strongly recommend consulting with a healthcare professional.';
            } else {
                // Low Risk
                titleElement.textContent = 'Low Risk';
                titleElement.className = 'risk-low';
                iconElement.textContent = '✅';
                messageElement.textContent = 'Based on the provided parameters, the model indicates a low risk of heart disease. Maintain a healthy lifestyle!';
            }
            
            renderChart(prob);
            
        } else {
            titleElement.textContent = 'Error';
            titleElement.className = 'risk-high';
            iconElement.textContent = '❌';
            messageElement.textContent = data.error || 'An unexpected error occurred.';
            document.querySelector('.chart-container').classList.add('hidden');
            document.querySelector('.probability-text').classList.add('hidden');
        }
        
        resetForm();
    }

    function renderChart(probability) {
        const ctx = document.getElementById('probabilityChart').getContext('2d');
        
        if (probabilityChart) {
            probabilityChart.destroy();
        }
        
        const isHighRisk = probability > 50;
        const color = isHighRisk ? '#ef4444' : '#10b981';
        
        probabilityChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Risk', 'Safe'],
                datasets: [{
                    data: [probability, 100 - probability],
                    backgroundColor: [
                        color,
                        'rgba(255, 255, 255, 0.1)'
                    ],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                cutout: '80%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                },
                animation: {
                    animateScale: true,
                    animateRotate: true
                }
            }
        });
        
        document.querySelector('.chart-container').classList.remove('hidden');
        document.querySelector('.probability-text').classList.remove('hidden');
    }
    
    function resetForm() {
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
        predictBtn.disabled = false;
    }
    
    // Add staggered animation to form groups
    const formGroups = document.querySelectorAll('.form-group');
    formGroups.forEach((group, index) => {
        group.style.opacity = '0';
        group.style.animation = `fadeInUp 0.5s ease-out forwards ${0.3 + (index * 0.05)}s`;
    });
});
