// ========== AUTHENTICATION SYSTEM ==========

let users = {};
let currentUser = null;

/**
 * Initialize auth system
 */
function initAuth() {
    users = parseJSON(localStorage.getItem('users') || '{}', {});
    const savedUser = localStorage.getItem('currentUser');
    
    if (savedUser) {
        currentUser = parseJSON(savedUser);
        showDashboard();
    } else {
        showAuth();
    }
}

/**
 * Toggle between login and signup
 */
function toggleAuthForm() {
    document.getElementById('loginForm').classList.toggle('hidden');
    document.getElementById('signupForm').classList.toggle('hidden');
    hideError('loginError');
    hideError('signupError');
}

/**
 * Handle login
 */
function handleLogin(e) {
    e.preventDefault();
    
    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value;
    
    if (!email || !password) {
        showError('loginError', 'Please fill all fields');
        return;
    }
    
    if (!users[email]) {
        showError('loginError', 'Email not found. Please sign up first.');
        return;
    }
    
    if (users[email].password !== password) {
        showError('loginError', 'Invalid password');
        return;
    }
    
    hideError('loginError');
    currentUser = { email, name: users[email].name };
    localStorage.setItem('currentUser', JSON.stringify(currentUser));
    
    log(`✅ Login successful: ${email}`);
    showDashboard();
}

/**
 * Handle signup
 */
function handleSignup(e) {
    e.preventDefault();
    
    const name = document.getElementById('signupName').value.trim();
    const email = document.getElementById('signupEmail').value.trim();
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('signupPasswordConfirm').value;
    
    if (!name || !email || !password || !confirmPassword) {
        showError('signupError', 'Please fill all fields');
        return;
    }
    
    if (!isValidEmail(email)) {
        showError('signupError', 'Please enter a valid email');
        return;
    }
    
    if (!isValidPassword(password)) {
        showError('signupError', 'Password must be at least 6 characters');
        return;
    }
    
    if (password !== confirmPassword) {
        showError('signupError', 'Passwords do not match');
        return;
    }
    
    if (users[email]) {
        showError('signupError', 'Email already registered');
        return;
    }
    
    hideError('signupError');
    users[email] = { name, password };
    localStorage.setItem('users', JSON.stringify(users));
    
    currentUser = { email, name };
    localStorage.setItem('currentUser', JSON.stringify(currentUser));
    
    log(`✅ Signup successful: ${name} (${email})`);
    showDashboard();
}

/**
 * Handle logout
 */
function handleLogout() {
    currentUser = null;
    localStorage.removeItem('currentUser');
    
    document.getElementById('loginEmail').value = '';
    document.getElementById('loginPassword').value = '';
    document.getElementById('signupName').value = '';
    document.getElementById('signupEmail').value = '';
    document.getElementById('signupPassword').value = '';
    document.getElementById('signupPasswordConfirm').value = '';
    
    document.getElementById('signupForm').classList.add('hidden');
    document.getElementById('loginForm').classList.remove('hidden');
    
    log('✅ Logged out');
    showAuth();
}

/**
 * Get current user
 */
function getCurrentUser() {
    return currentUser;
}

/**
 * Check if authenticated
 */
function isAuthenticated() {
    return currentUser !== null;
}
