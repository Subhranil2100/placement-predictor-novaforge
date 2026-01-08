// ========== ML PREDICTION ENGINE ==========

/**
 * Predict placement using Random Forest-inspired algorithm
 */
function predictPlacement(studentData) {
    // Feature weights
    const weights = {
        cgpa: 0.25,
        marks: 0.20,
        experience: 0.25,
        communication: 0.15,
        backlogs: 0.15
    };
    
    // Calculate feature scores (0-1 range)
    const cgpaScore = studentData.cgpa / 10;
    const marksScore = ((studentData.marks10 + studentData.marks12) / 2) / 100;
    const experienceScore = (
        (studentData.internships * 15 + studentData.projects * 8) / 100
    );
    const communicationScore = studentData.communication / 5;
    const backlogScore = 1 - (studentData.backlogs / 10);
    
    // Calculate weighted score
    const baseScore = 
        (cgpaScore * weights.cgpa) +
        (marksScore * weights.marks) +
        (Math.min(experienceScore, 1) * weights.experience) +
        (communicationScore * weights.communication) +
        (Math.max(backlogScore, 0) * weights.backlogs);
    
    // Placement probability (clamp to 0.15-0.98)
    const placementProbability = Math.min(0.98, Math.max(0.15, baseScore));
    const isPlaced = placementProbability > 0.5;
    
    // Determine company tier
    let companyTier = 'Not Placed';
    
    if (isPlaced) {
        const tierScore = 
            (cgpaScore * 0.4) + 
            (marksScore * 0.3) + 
            (experienceScore * 0.3);
        
        if (tierScore > 0.8) {
            companyTier = 'Tier 1 - Google, Microsoft, Amazon';
        } else if (tierScore > 0.65) {
            companyTier = 'Tier 2 - TCS, Infosys, Wipro';
        } else if (tierScore > 0.5) {
            companyTier = 'Tier 3 - Tech Mahindra, HCL, IBM';
        } else {
            companyTier = 'Startups & SMEs';
        }
    }
    
    // Calculate feature importance
    const features = {
        'CGPA': (cgpaScore * 100).toFixed(1),
        'Academic Marks': (marksScore * 100).toFixed(1),
        'Internships': Math.min(100, (studentData.internships * 20)).toFixed(1),
        'Projects': Math.min(100, (studentData.projects * 10)).toFixed(1),
        'Communication': (communicationScore * 100).toFixed(1),
        'No Backlogs': Math.min(100, (backlogScore * 100)).toFixed(1)
    };
    
    return {
        placed: isPlaced,
        probability: (placementProbability * 100).toFixed(1),
        companyTier,
        features
    };
}

/**
 * NEW: Predict sector using ML backend
 */
async function predictSector(studentData) {
    try {
        // Map student data to ML features (matching backend)
        const sectorPayload = {
            score800: studentData.cgpa * 80,
            aptitude: ((studentData.marks10 + studentData.marks12) / 2) * 0.7,
            english: studentData.marks12 * 0.8,
            quantitative: ((studentData.marks10 + studentData.marks12) / 2) * 0.75,
            analytical: studentData.cgpa * 10,
            domain: studentData.projects * 5,
            comp_fund: studentData.cgpa * 8,
            coding: studentData.projects * 4,
            personality: studentData.communication * 20
        };

        const response = await fetch('http://localhost:5001/api/sector-predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(sectorPayload)
        });

        if (!response.ok) {
            throw new Error('Sector prediction API failed');
        }

        const data = await response.json();
        return {
            success: true,
            bestSector: data.best_sector,
            bestProbability: data.best_probability,
            top3: data.top_3,
            allSectors: data.all_sectors,
            message: data.message
        };
    } catch (error) {
        console.warn('Sector API unavailable:', error);
        return {
            success: false,
            message: 'Sector prediction unavailable. Check backend.',
            error: error.message
        };
    }
}

/**
 * Validate student data
 */
function validateStudentData(data) {
    const errors = [];
    
    if (!data.rollNumber) errors.push('Roll Number is required');
    if (!data.branch) errors.push('Branch is required');
    if (data.marks10 < 0 || data.marks10 > 100) errors.push('10th marks must be 0-100');
    if (data.marks12 < 0 || data.marks12 > 100) errors.push('12th marks must be 0-100');
    if (data.cgpa < 0 || data.cgpa > 10) errors.push('CGPA must be 0-10');
    if (data.internships < 0 || data.internships > 5) errors.push('Internships must be 0-5');
    if (data.projects < 0 || data.projects > 20) errors.push('Projects must be 0-20');
    if (data.communication < 1 || data.communication > 5) errors.push('Communication must be 1-5');
    if (data.backlogs < 0 || data.backlogs > 10) errors.push('Backlogs must be 0-10');
    
    return errors;
}
