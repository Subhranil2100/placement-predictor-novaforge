// ========== MAIN APPLICATION LOGIC ==========

/**
 * Handle prediction form submission
 */
async function handlePrediction(event) {
    event.preventDefault();
    
    // Get form data
    const studentData = {
        rollNumber: document.getElementById('rollNumber').value,
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

    // Validate data
    const errors = validateStudentData(studentData);
    if (errors.length > 0) {
        showError(errors.join(', '));
        return;
    }

    // Show loading state
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = '⏳ Analyzing...';
    submitBtn.disabled = true;

    try {
        // 1. Placement prediction (local)
        const placementResult = predictPlacement(studentData);
        displayPlacementResults(placementResult, studentData);

        // 2. Sector prediction (API)
        const sectorResult = await predictSector(studentData);
        displaySectorResults(sectorResult);

        // Show success message
        showSuccess(`✅ Prediction completed for ${studentData.rollNumber}!`);

        // Scroll to results
        setTimeout(() => {
            document.getElementById('predictionResult').scrollIntoView({ behavior: 'smooth' });
        }, 300);

    } catch (error) {
        showError('Prediction failed: ' + error.message);
        console.error('Prediction error:', error);
    } finally {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
}

/**
 * Display placement prediction results
 */
function displayPlacementResults(result, studentData) {
    const resultDiv = document.getElementById('predictionResult');
    
    // Show results section
    resultDiv.style.display = 'block';
    
    // Update placement status
    const statusText = result.placed ? '✅ LIKELY TO BE PLACED' : '⚠️ AT RISK';
    const statusColor = result.placed ? '#27ae60' : '#e74c3c';
    document.getElementById('placementStatus').textContent = statusText;
    document.getElementById('placementStatus').style.color = statusColor;
    
    // Update confidence score
    document.getElementById('confidenceScore').textContent = result.probability + '%';
    document.getElementById('confidenceFill').style.width = result.probability + '%';
    document.getElementById('confidenceFill').style.backgroundColor = statusColor;
    
    // Update company tier
    document.getElementById('companyTier').textContent = result.companyTier;
    
    // Display feature importance
    displayFeatureImportance(result.features);
}

/**
 * Display sector prediction results
 */
function displaySectorResults(sectorData) {
    const sectorResult = document.getElementById('sectorResult');
    
    if (!sectorData.success) {
        console.warn('Sector prediction unavailable');
        sectorResult.style.display = 'none';
        return;
    }

    // Show sector section
    sectorResult.style.display = 'block';
    
    // Best sector
    document.getElementById('bestSectorName').textContent = sectorData.bestSector;
    document.getElementById('bestSectorProb').textContent = `Fit Score: ${sectorData.bestProbability}`;

    // Top 3 sectors
    const top3HTML = sectorData.top3.map((item, i) => `
        <div style="background: var(--color-surface); padding: 18px; border-radius: 10px; border-left: 4px solid var(--color-primary); box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <div style="font-size: 12px; color: var(--color-text-secondary); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Rank ${i + 1}</div>
                    <div style="font-weight: 700; color: var(--color-text); margin-top: 6px; font-size: 16px;">${item.sector}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 20px; font-weight: 700; color: var(--color-primary);">${item.probability}</div>
                    <div style="font-size: 12px; color: var(--color-text-secondary); margin-top: 2px;">Fit Score</div>
                </div>
            </div>
            <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--color-border); font-size: 13px; color: var(--color-text-secondary);">
                ${item.recommendation}
            </div>
        </div>
    `).join('');
    document.getElementById('top3SectorsContainer').innerHTML = top3HTML;

    // All sectors ranked
    const allHTML = sectorData.allSectors.map((item, index) => {
        const fitValue = parseFloat(item.probability);
        let barColor = 'var(--color-primary)';
        if (fitValue < 50) barColor = 'var(--color-warning)';
        if (fitValue < 40) barColor = 'var(--color-error)';
        
        return `
        <div style="display: flex; align-items: center; padding: 12px; background: var(--color-bg); border-radius: 8px; border: 1px solid var(--color-border);">
            <div style="flex: 0 0 30px; text-align: center; font-weight: 600; color: var(--color-primary);">${index + 1}</div>
            <div style="flex: 1; padding: 0 15px;">
                <div style="font-weight: 600; color: var(--color-text); font-size: 14px;">${item.sector}</div>
                <div style="font-size: 12px; color: var(--color-text-secondary); margin-top: 3px;">${item.recommendation}</div>
            </div>
            <div style="flex: 0 0 100px; text-align: right;">
                <div style="display: inline-block; background: ${barColor}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600;">
                    ${item.probability}
                </div>
            </div>
        </div>
        `;
    }).join('');
    document.getElementById('allSectorsContainer').innerHTML = allHTML;
}

/**
 * Display feature importance
 */
function displayFeatureImportance(features) {
    const featuresSection = document.getElementById('featuresSection');
    const featuresContainer = document.getElementById('featuresContainer');
    
    // Show features section
    featuresSection.style.display = 'block';
    
    // Create feature bars
    const featureHTML = Object.entries(features).map(([name, value]) => {
        const numValue = parseFloat(value);
        const barColor = numValue > 75 ? '#27ae60' : numValue > 50 ? '#f39c12' : '#e74c3c';
        
        return `
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-weight: 600; color: var(--color-text);">${name}</span>
                <span style="font-weight: 600; color: ${barColor};">${value}%</span>
            </div>
            <div style="background: var(--color-border); height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: ${barColor}; height: 100%; width: ${numValue}%; border-radius: 4px; transition: width 0.6s ease-out;"></div>
            </div>
        </div>
        `;
    }).join('');
    
    featuresContainer.innerHTML = featureHTML;
}

/**
 * Show error message
 */
function showError(message) {
    const errorEl = document.getElementById('successMessage');
    errorEl.innerHTML = `<div class="error" style="padding: 16px; background: #ffebee; border: 1px solid #ef5350; border-radius: 8px; color: #c62828; margin-bottom: 20px;">❌ ${message}</div>`;
    errorEl.style.display = 'block';
    setTimeout(() => {
        errorEl.style.display = 'none';
    }, 5000);
}

/**
 * Show success message
 */
function showSuccess(message) {
    const successEl = document.getElementById('successMessage');
    successEl.innerHTML = `<div style="padding: 16px; background: #e8f5e9; border: 1px solid #66bb6a; border-radius: 8px; color: #2e7d32; margin-bottom: 20px;">${message}</div>`;
    successEl.style.display = 'block';
    setTimeout(() => {
        successEl.style.display = 'none';
    }, 5000);
}

/**
 * Handle login
 */
function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    
    // Simple validation
    if (!email || !password) {
        document.getElementById('loginError').textContent = 'Email and password are required';
        return;
    }
    
    // Store user info (localStorage)
    const user = {
        email,
        name: email.split('@')[0],
        timestamp: new Date().toISOString()
    };
    
    localStorage.setItem('currentUser', JSON.stringify(user));
    
    // Show dashboard
    document.getElementById('authSection').classList.add('hidden');
    document.getElementById('dashboardSection').classList.remove('hidden');
    document.getElementById('userDisplay').textContent = user.name;
    
    showSuccess(`Welcome back, ${user.name}!`);
}

/**
 * Handle signup
 */
function handleSignup(event) {
    event.preventDefault();
    
    const name = document.getElementById('signupName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const passwordConfirm = document.getElementById('signupPasswordConfirm').value;
    
    // Validate
    if (!name || !email || !password) {
        document.getElementById('signupError').textContent = 'All fields are required';
        return;
    }
    
    if (password !== passwordConfirm) {
        document.getElementById('signupError').textContent = 'Passwords do not match';
        return;
    }
    
    if (password.length < 6) {
        document.getElementById('signupError').textContent = 'Password must be at least 6 characters';
        return;
    }
    
    // Store user
    const user = {
        name,
        email,
        timestamp: new Date().toISOString()
    };
    
    localStorage.setItem('currentUser', JSON.stringify(user));
    
    // Show dashboard
    document.getElementById('authSection').classList.add('hidden');
    document.getElementById('dashboardSection').classList.remove('hidden');
    document.getElementById('userDisplay').textContent = name;
    
    showSuccess(`Welcome, ${name}! Account created successfully.`);
}

/**
 * Handle logout
 */
function handleLogout() {
    localStorage.removeItem('currentUser');
    document.getElementById('dashboardSection').classList.add('hidden');
    document.getElementById('authSection').classList.remove('hidden');
    document.getElementById('loginForm').classList.remove('hidden');
    document.getElementById('signupForm').classList.add('hidden');
    document.getElementById('predictionResult').style.display = 'none';
    document.getElementById('sectorResult').style.display = 'none';
    document.getElementById('featuresSection').style.display = 'none';
}

/**
 * Toggle between login and signup
 */
function toggleAuthForm() {
    document.getElementById('loginForm').classList.toggle('hidden');
    document.getElementById('signupForm').classList.toggle('hidden');
    document.getElementById('loginError').textContent = '';
    document.getElementById('signupError').textContent = '';
}

/**
 * Initialize app on page load
 */
document.addEventListener('DOMContentLoaded', function() {
    // Check if user is logged in
    const user = JSON.parse(localStorage.getItem('currentUser') || 'null');
    
    if (user) {
        document.getElementById('authSection').classList.add('hidden');
        document.getElementById('dashboardSection').classList.remove('hidden');
        document.getElementById('userDisplay').textContent = user.name;
    }
});