// ========== UI MANAGEMENT ==========

/**
 * Show auth section
 */
function showAuth() {
    document.getElementById('authSection').classList.remove('hidden');
    document.getElementById('dashboardSection').classList.add('hidden');
}

/**
 * Show dashboard section
 */
function showDashboard() {
    document.getElementById('authSection').classList.add('hidden');
    document.getElementById('dashboardSection').classList.remove('hidden');
    
    if (currentUser) {
        const firstName = currentUser.name.split(' ')[0];
        document.getElementById('userDisplay').textContent = firstName;
    }
}

/**
 * Display prediction results
 */
function displayPredictionResults(prediction) {
    document.getElementById('placementStatus').textContent = 
        prediction.placed ? '✓ Placed' : '✗ Not Placed';
    document.getElementById('confidenceScore').textContent = 
        prediction.probability + '%';
    document.getElementById('confidenceFill').style.width = 
        prediction.probability + '%';
    document.getElementById('companyTier').innerHTML = 
        `<span class="company-badge">${prediction.companyTier}</span>`;
}

/**
 * Display feature importance
 */
function displayFeatureImportance(features) {
    const container = document.getElementById('featuresContainer');
    container.innerHTML = '';
    
    Object.entries(features).forEach(([name, value]) => {
        const numValue = parseFloat(value);
        const html = `
            <div class="feature-item">
                <div class="feature-name">${name}</div>
                <div class="feature-bar">
                    <div class="feature-fill" style="width: ${numValue}%"></div>
                </div>
                <div class="feature-percent">${value}%</div>
            </div>
        `;
        container.innerHTML += html;
    });
}

/**
 * Show prediction result
 */
function showPredictionResult() {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.classList.add('show');
    document.getElementById('featuresSection').classList.add('show');
    
    setTimeout(() => {
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Get student data from form
 */
function getStudentDataFromForm() {
    return {
        rollNumber: document.getElementById('rollNumber').value.trim(),
        branch: document.getElementById('branch').value,
        marks10: parseFloat(document.getElementById('marks10').value),
        marks12: parseFloat(document.getElementById('marks12').value),
        cgpa: parseFloat(document.getElementById('cgpa').value),
        internships: parseInt(document.getElementById('internships').value),
        projects: parseInt(document.getElementById('projects').value),
        communication: parseFloat(document.getElementById('communication').value),
        backlogs: parseInt(document.getElementById('backlogs').value),
        hostel: document.getElementById('hostel').value
    };
}

/**
 * Clear prediction results
 */
function clearPredictionResults() {
    document.getElementById('placementStatus').textContent = '—';
    document.getElementById('confidenceScore').textContent = '—';
    document.getElementById('confidenceFill').style.width = '0%';
    document.getElementById('companyTier').innerHTML = '—';
    document.getElementById('featuresContainer').innerHTML = '';
}

/**
 * Reset form
 */
function resetPredictionForm() {
    document.getElementById('rollNumber').value = '';
    document.getElementById('branch').value = '';
    document.getElementById('marks10').value = '';
    document.getElementById('marks12').value = '';
    document.getElementById('cgpa').value = '';
    document.getElementById('internships').value = '';
    document.getElementById('projects').value = '';
    document.getElementById('communication').value = '';
    document.getElementById('backlogs').value = '';
    document.getElementById('hostel').value = '';
}
