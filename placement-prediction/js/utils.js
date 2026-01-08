// ========== UTILITY FUNCTIONS ==========

/**
 * Show/hide element
 */
function toggleVisibility(elementId) {
    const element = document.getElementById(elementId);
    element.classList.toggle('hidden');
}

/**
 * Hide element
 */
function hideElement(elementId) {
    document.getElementById(elementId).classList.add('hidden');
}

/**
 * Show element
 */
function showElement(elementId) {
    document.getElementById(elementId).classList.remove('hidden');
}

/**
 * Display error message
 */
function showError(elementId, message) {
    const errorDiv = document.getElementById(elementId);
    errorDiv.textContent = message;
    errorDiv.classList.add('show');
}

/**
 * Hide error message
 */
function hideError(elementId) {
    const errorDiv = document.getElementById(elementId);
    errorDiv.classList.remove('show');
}

/**
 * Display success message
 */
function showSuccess(message) {
    const successMsg = document.getElementById('successMessage');
    successMsg.textContent = message;
    successMsg.classList.add('show');
    setTimeout(() => successMsg.classList.remove('show'), 5000);
}

/**
 * Validate email
 */
function isValidEmail(email) {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
}

/**
 * Validate password length
 */
function isValidPassword(password) {
    return password && password.length >= 6;
}

/**
 * Parse JSON safely
 */
function parseJSON(jsonString, defaultValue = null) {
    try {
        return JSON.parse(jsonString);
    } catch (error) {
        console.error('JSON parse error:', error);
        return defaultValue;
    }
}

/**
 * Log with timestamp
 */
function log(message) {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] ${message}`);
}
