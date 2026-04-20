"""
Display-only mappings for the FinSight AI Streamlit application.

These dictionaries map the exact raw values expected by the model and API 
to human-readable labels for the Streamlit UI. Keys should never be changed 
as they correspond directly to dataset features.
"""

EDUCATION_LABELS = {
    "illiterate":         "No formal education",
    "basic.4y":           "Primary school (4 years)",
    "basic.6y":           "Middle school (6 years)",
    "basic.9y":           "Lower secondary (9 years)",
    "high.school":        "High school",
    "professional.course":"Professional course",
    "university.degree":  "University degree",
    "unknown":            "Unknown / Not disclosed",
}

JOB_LABELS = {
    "admin.":        "Admin",
    "blue-collar":   "Blue-collar",
    "entrepreneur":  "Entrepreneur",
    "housemaid":     "Housemaid",
    "management":    "Management",
    "retired":       "Retired",
    "self-employed": "Self-employed",
    "services":      "Services",
    "student":       "Student",
    "technician":    "Technician",
    "unemployed":    "Unemployed",
    "unknown":       "Unknown",
}

MARITAL_LABELS = {
    "married":  "Married",
    "single":   "Single",
    "divorced": "Divorced / Widowed",
    "unknown":  "Not disclosed",
}

YES_NO_UNKNOWN = {
    "no":      "No",
    "yes":     "Yes",
    "unknown": "Not disclosed",
}

CONTACT_LABELS = {
    "cellular":   "Mobile phone",
    "telephone":  "Landline",
}

MONTH_LABELS = {
    "jan": "January",  "feb": "February", "mar": "March",
    "apr": "April",    "may": "May",       "jun": "June",
    "jul": "July",     "aug": "August",    "sep": "September",
    "oct": "October",  "nov": "November",  "dec": "December",
}

DAY_LABELS = {
    "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
    "thu": "Thursday", "fri": "Friday",
}

POUTCOME_LABELS = {
    "nonexistent": "No previous campaign",
    "failure":     "Previous campaign failed",
    "success":     "Previous campaign succeeded",
}