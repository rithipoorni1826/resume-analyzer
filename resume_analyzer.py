# Install required packages (run this cell first)
# !pip install nltk spacy docx2txt PyPDF2 pandas scikit-learn pdfplumber
# !python -m spacy download en_core_web_md

import os
import re
import nltk
import spacy
import docx2txt
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pdfplumber
from IPython.display import display, HTML

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_md')
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load('en_core_web_md')

def preprocess_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    
    return text

def extract_text_from_pdf(file_path):
    """Extract text from PDF file."""
    text = ""
    try:
        # Try using PyPDF2 first
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
        
        # If PyPDF2 returns empty or minimal text, try pdfplumber
        if len(text.strip()) < 100:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or "" + " "
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file."""
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_skills(text):
    """Extract skills from text using spaCy NER and predefined skill list."""
    skills = []
    
    # Common technical and soft skills
    skill_patterns = [
        "python", "javascript", "java", "c\\+\\+", "c#", "sql", "nosql", "mongodb",
        "react", "angular", "vue", "node.js", "express", "django", "flask", "ruby",
        "php", "html", "css", "aws", "azure", "gcp", "docker", "kubernetes",
        "machine learning", "deep learning", "artificial intelligence", "data science",
        "data analysis", "statistics", "excel", "tableau", "power bi", "tensorflow",
        "pytorch", "scikit-learn", "pandas", "numpy", "r", "matlab", "scala", "hadoop",
        "spark", "kafka", "redux", "typescript", "git", "github", "gitlab", "ci/cd",
        "jenkins", "agile", "scrum", "kanban", "jira", "confluence", "leadership",
        "communication", "teamwork", "problem solving", "critical thinking",
        "project management", "time management", "creativity", "collaboration",
        "adaptability", "flexibility", "organization", "presentation", "negotiation"
    ]
    
    # Create a pattern to search for skills
    pattern = r'\b(' + '|'.join(skill_patterns) + r')\b'
    
    # Find all matches
    matches = re.finditer(pattern, text.lower())
    for match in matches:
        skill = match.group(0)
        if skill not in skills:
            skills.append(skill)
    
    # Use spaCy for additional skill extraction (especially multi-word skills)
    doc = nlp(text)
    
    # Look for noun phrases that might be skills
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        # Check if the chunk contains skill-related words
        if any(skill in chunk_text for skill in ["experience", "proficient", "knowledge", "skill", "expert"]):
            # Extract the skill part
            for token in chunk:
                if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in stopwords.words('english'):
                    if token.text.lower() not in skills:
                        skills.append(token.text.lower())
    
    return skills

def extract_education(text):
    """Extract education information using regex patterns."""
    education = []
    
    # Education degree patterns
    degree_patterns = [
        r'\b(?:Ph\.?D\.?|Doctor of Philosophy)\b',
        r'\bM\.?S\.?|Master of Science\b',
        r'\bM\.?A\.?|Master of Arts\b',
        r'\bM\.?B\.?A\.?|Master of Business Administration\b',
        r'\bB\.?S\.?|Bachelor of Science\b',
        r'\bB\.?A\.?|Bachelor of Arts\b',
        r'\bB\.?Tech\.?|Bachelor of Technology\b',
        r'\bM\.?Tech\.?|Master of Technology\b',
        r'\bAssociate Degree\b',
        r'\bHigh School Diploma\b'
    ]
    
    for pattern in degree_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get surrounding context (50 characters before and after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            if context not in education:
                education.append(context)
    
    return education

def extract_experience(text):
    """Extract work experience information using regex patterns."""
    experience = []
    
    # Look for job titles, dates, and company names
    job_title_pattern = r'\b(?:Senior|Junior|Lead|Chief|Principal|Director|Manager|Engineer|Developer|Analyst|Consultant|Specialist|Coordinator|Administrator|Assistant|Associate|Supervisor|Officer|Executive|VP|President|CEO|CTO|CFO)\b'
    
    matches = re.finditer(job_title_pattern, text)
    for match in matches:
        # Get surrounding context (100 characters before and after)
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        
        if context not in experience:
            experience.append(context)
    
    # Also look for date patterns that might indicate employment periods
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\s+(?:to|–|-)\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\s+(?:to|–|-)\s+[Pp]resent\b'
    
    matches = re.finditer(date_pattern, text)
    for match in matches:
        # Get surrounding context (100 characters before and after)
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        
        if context not in experience:
            experience.append(context)
    
    return experience

def calculate_skill_match_score(resume_skills, job_skills):
    """Calculate skill match score based on common skills."""
    if not job_skills:
        return 0, []
    
    common_skills = set(resume_skills).intersection(set(job_skills))
    score = len(common_skills) / len(job_skills) * 100
    return score, list(common_skills)

def calculate_text_similarity(resume_text, job_description):
    """Calculate text similarity using cosine similarity."""
    # Initialize CountVectorizer
    vectorizer = CountVectorizer()
    
    # Create document-term matrix
    try:
        count_matrix = vectorizer.fit_transform([resume_text, job_description])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(count_matrix)[0][1] * 100
        return similarity
    except:
        return 0

def extract_job_requirements(job_description):
    """Extract key skills and requirements from job description."""
    skills = extract_skills(job_description)
    
    # Look for requirement sections
    requirement_section = re.search(r'(?:Requirements|Qualifications|Skills Required|What You Need)(?:[\s\:\-]+)(.*?)(?:\n\n|\Z)', 
                                   job_description, re.IGNORECASE | re.DOTALL)
    
    requirements = []
    if requirement_section:
        req_text = requirement_section.group(1)
        # Split by bullet points or new lines
        for line in re.split(r'•|\*|\n', req_text):
            if line.strip():
                requirements.append(line.strip())
    
    return skills, requirements

def analyze_resume(resume_text, job_description):
    """Analyze resume against job description and provide scoring."""
    # Preprocess texts
    processed_resume = preprocess_text(resume_text)
    processed_job = preprocess_text(job_description)
    
    # Extract information
    resume_skills = extract_skills(resume_text)
    job_skills, job_requirements = extract_job_requirements(job_description)
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)
    
    # Calculate scores
    skill_score, matching_skills = calculate_skill_match_score(resume_skills, job_skills)
    content_similarity = calculate_text_similarity(processed_resume, processed_job)
    
    # Overall score calculation (weighted average)
    overall_score = (skill_score * 0.6) + (content_similarity * 0.4)
    
    return {
        'overall_score': round(overall_score, 2),
        'skill_score': round(skill_score, 2),
        'content_similarity': round(content_similarity, 2),
        'resume_skills': resume_skills,
        'job_skills': job_skills,
        'matching_skills': matching_skills,
        'education': education,
        'experience': experience,
        'job_requirements': job_requirements
    }

def generate_recommendations(analysis):
    """Generate improvement recommendations based on analysis."""
    recommendations = []
    
    # Check skill match
    skill_score = analysis['skill_score']
    if skill_score < 50:
        missing_skills = set(analysis['job_skills']) - set(analysis['resume_skills'])
        recommendations.append(f"Your resume matches only {skill_score:.2f}% of the required skills. Consider highlighting or developing these missing skills: {', '.join(list(missing_skills)[:5])}.")
    
    # Check content similarity
    content_score = analysis['content_similarity']
    if content_score < 40:
        recommendations.append("Your resume content doesn't strongly align with the job description. Try tailoring your experience descriptions to better match the job requirements.")
    
    # Check education section
    if not analysis['education']:
        recommendations.append("Education details weren't clearly detected. Ensure your education section is well-formatted and comprehensive.")
    
    # Check experience section
    if len(analysis['experience']) < 2:
        recommendations.append("Work experience details seem limited. Make sure your experience section is detailed with clear job titles, companies, and dates.")
    
    # If all scores are good
    if skill_score >= 70 and content_score >= 60 and analysis['education'] and len(analysis['experience']) >= 2:
        recommendations.append("Your resume is well-aligned with this job position! Consider emphasizing your most relevant achievements even more prominently.")
    
    return recommendations

# Function to display results with HTML formatting
def display_results(results, recommendations):
    # Overall score
    score = results['overall_score']
    if score >= 75:
        score_color = "green"
    elif score >= 50:
        score_color = "orange"
    else:
        score_color = "red"
    
    html = f"""
    <h2>Analysis Results</h2>
    <h1 style='text-align: center; color: {score_color};'>{score}%</h1>
    
    <h3>Score Breakdown</h3>
    <table style='width:50%'>
        <tr>
            <td><b>Skills Match:</b></td>
            <td>{results['skill_score']}%</td>
        </tr>
        <tr>
            <td><b>Content Similarity:</b></td>
            <td>{results['content_similarity']}%</td>
        </tr>
    </table>
    
    <h3>Skills Analysis</h3>
    <table style='width:100%'>
        <tr>
            <th>Your Skills</th>
            <th>Required Skills</th>
            <th>Matching Skills</th>
        </tr>
        <tr>
            <td style='vertical-align:top'>
    """
    
    # Your skills
    if results['resume_skills']:
        for skill in results['resume_skills']:
            html += f"• {skill}<br>"
    else:
        html += "No skills detected<br>"
    
    html += "</td><td style='vertical-align:top'>"
    
    # Required skills
    if results['job_skills']:
        for skill in results['job_skills']:
            html += f"• {skill}<br>"
    else:
        html += "No skills detected in job description<br>"
    
    html += "</td><td style='vertical-align:top'>"
    
    # Matching skills
    if results['matching_skills']:
        for skill in results['matching_skills']:
            html += f"• {skill}<br>"
    else:
        html += "No matching skills found<br>"
    
    html += """
        </td>
        </tr>
    </table>
    
    <h3>Education Detected</h3>
    """
    
    if results['education']:
        for edu in results['education']:
            html += f"• {edu}<br>"
    else:
        html += "No education details detected<br>"
    
    html += "<h3>Experience Detected</h3>"
    
    if results['experience']:
        for exp in results['experience'][:3]:  # Show top 3
            html += f"• {exp}<br>"
    else:
        html += "No experience details detected<br>"
    
    html += "<h3>Recommendations</h3>"
    
    for rec in recommendations:
        html += f"• {rec}<br>"
    
    display(HTML(html))
    
    # Create and save DataFrame
    data = {
        "Metric": ["Overall Score", "Skills Match", "Content Similarity", 
                   "Your Skills", "Required Skills", "Matching Skills",
                   "Recommendations"],
        "Value": [
            f"{results['overall_score']}%",
            f"{results['skill_score']}%",
            f"{results['content_similarity']}%",
            ", ".join(results['resume_skills']),
            ", ".join(results['job_skills']),
            ", ".join(results['matching_skills']),
            "; ".join(recommendations)
        ]
    }
    df = pd.DataFrame(data)
    return df

# RUN THIS CELL TO ANALYZE A RESUME AGAINST A JOB DESCRIPTION
# =========================================================
# Step 1: Set the path to your resume file
resume_path = "Resume.pdf"  # CHANGE THIS to your actual file path

# Step 2: Paste the job description
job_description = "Develop, test and maintain high-quality software using Python programming language. Participate in the entire software development lifecycle, building, testing and delivering high-quality solutions. Collaborate with cross-functional teams to identify and solve complex problems."
# REPLACE with actual job description

# Step 3: Extract text from resume
print("Extracting text from resume...")
if resume_path.lower().endswith('.pdf'):
    resume_text = extract_text_from_pdf(resume_path)
elif resume_path.lower().endswith('.docx'):
    resume_text = extract_text_from_docx(resume_path)
else:
    print("Unsupported file format. Please use PDF or DOCX.")
    resume_text = ""

if resume_text:
    print(f"Successfully extracted {len(resume_text)} characters from resume.")
    
    # Print first 200 characters to verify extraction
    print("Preview of extracted text:")
    print(resume_text[:200] + "...")
    
    # Step 4: Analyze resume
    print("\nAnalyzing resume against job description...")
    analysis_results = analyze_resume(resume_text, job_description)
    recommendations = generate_recommendations(analysis_results)
    
    # Step 5: Display and save results
    print("\nResults:")
    results_df = display_results(analysis_results, recommendations)
    
    # Optional: Save to CSV
    # results_df.to_csv('resume_analysis.csv')
    print("\nAnalysis complete. Use results_df.to_csv('filename.csv') to save results.")
else:
    print("Failed to extract text from resume. Please check the file path and format.")
