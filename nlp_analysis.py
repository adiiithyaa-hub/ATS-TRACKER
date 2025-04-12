import streamlit as st
import os
import json
import anthropic
import pandas as pd
import time
import PyPDF2
import docx
from typing import Dict, List, Tuple, Any, Union
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
import spacy
import matplotlib.pyplot as plt
import signal
import psutil

# Initialize NLP components
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Try loading spaCy model or download a smaller one if not available
try:
    nlp = spacy.load("en_core_web_md")
except:
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

class SemanticAnalyzer:
    """
    Class for performing semantic analysis between text documents.
    """
    
    def __init__(self):
        """Initialize the semantic analyzer"""
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for semantic analysis
        
        Args:
            text: Raw text string
            
        Returns:
            Processed text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Rejoin into string
        return ' '.join(filtered_tokens)
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract potential skills from text using NLP
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List of extracted skills
        """
        doc = nlp(text)
        
        # Extract noun phrases and named entities as potential skills
        skills = []
        
        # Get noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:  # Limit to reasonable skill name length
                skills.append(chunk.text.lower())
        
        # Add named entities that might be technologies or skills
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "GPE", "WORK_OF_ART"]:
                skills.append(ent.text.lower())
        
        # Add technical terms that might be important
        technical_patterns = [
            "python", "java", "javascript", "react", "node", "sql", "nosql", "aws", "azure", 
            "cloud", "docker", "kubernetes", "agile", "scrum", "ci/cd", "machine learning",
            "data science", "artificial intelligence", "ai", "ml", "deep learning", "nlp",
            "frontend", "backend", "full stack", "devops"
        ]
        
        for pattern in technical_patterns:
            if pattern in text.lower():
                skills.append(pattern)
        
        # Remove duplicates and sort
        return sorted(list(set(skills)))
    
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using TF-IDF and cosine similarity
        
        Args:
            text1: First text document
            text2: Second text document
            
        Returns:
            Similarity score between 0 and 1
        """
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def calculate_spacy_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using spaCy's vector representations
        
        Args:
            text1: First text document
            text2: Second text document
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            doc1 = nlp(text1)
            doc2 = nlp(text2)
            
            if doc1.vector_norm and doc2.vector_norm:
                return doc1.similarity(doc2)
            return 0.0
        except:
            return 0.0
    
    def calculate_skill_overlap(self, skills_jd: List[str], skills_resume: List[str]) -> Tuple[float, List[str], List[str]]:
        """
        Calculate skill overlap between job description and resume
        
        Args:
            skills_jd: List of skills from job description
            skills_resume: List of skills from resume
            
        Returns:
            Tuple containing overlap score, matched skills, and missing skills
        """
        skills_jd_lower = [s.lower() for s in skills_jd]
        skills_resume_lower = [s.lower() for s in skills_resume]
        
        # Find direct matches
        direct_matches = [skill for skill in skills_resume_lower if skill in skills_jd_lower]
        
        # Semantic matches (for skills that aren't direct matches)
        potential_semantic_matches = []
        missing_skills = []
        
        for jd_skill in skills_jd_lower:
            if jd_skill in direct_matches:
                continue
                
            # Calculate similarity with each resume skill
            best_match = None
            best_score = 0
            
            for resume_skill in skills_resume_lower:
                if resume_skill in direct_matches:
                    continue
                    
                # Calculate similarity
                doc1 = nlp(jd_skill)
                doc2 = nlp(resume_skill)
                
                try:
                    similarity = doc1.similarity(doc2)
                    
                    if similarity > 0.97 and similarity > best_score:  # Threshold for semantic match
                        best_match = resume_skill
                        best_score = similarity
                except:
                    continue
            
            if best_match:
                potential_semantic_matches.append((jd_skill, best_match, best_score))
            else:
                missing_skills.append(jd_skill)
        
        # Calculate overlap score
        matched_count = len(direct_matches) + len(potential_semantic_matches)
        total_required = len(skills_jd_lower)
        
        if total_required == 0:
            overlap_score = 0.0
        else:
            overlap_score = matched_count / total_required * 100
            
        # Create matched skills list with both direct and semantic matches
        matched_skills = direct_matches + [f"{m[0]} (similar to '{m[1]}')" for m in potential_semantic_matches]
        
        return overlap_score, matched_skills, missing_skills
        
class JDResumeMatcher:
    """
    A class that uses Claude API and semantic analysis to calculate matching scores between job descriptions and resumes.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the JDResumeMatcher with API key.
        
        Args:
            api_key: Claude API key (defaults to environment variable)
        """
        self.api_key = api_key or os.environ.get("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")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.semantic_analyzer = SemanticAnalyzer()
        
        # System prompt that instructs Claude on how to analyze and score
        self.system_prompt = """
        You are an expert ATS (Applicant Tracking System) analyzer that evaluates the match between job descriptions and resumes.
        Your task is to carefully analyze both documents and provide a detailed assessment of how well the candidate's resume matches the job requirements.
        
        Follow these steps:
        
        1. Thoroughly extract and analyze key information from both the job description and resume, including:
           - Required skills and technologies
           - Years of experience
           - Education requirements
           - Industry knowledge
           - Certifications and qualifications
           - Soft skills and competencies
           - Specific domain expertise
        
        2. Identify and highlight:
           - Direct matches between the resume and job description
           - Partial matches where the candidate has related but not exact skills/experience
           - Missing requirements that appear in the JD but not in the resume
           - Standout qualifications the candidate has that exceed expectations
           - Notice Period and Salary expectations
           -- Evidence of achievements and impact in previous roles
        
        3.Compare the Eligibility Criteria and Requirements of job description with the resume and score accordingly a match score from 0-100 based on:
           - Required skills coverage (55% weight) 
             * Technical skills (30%)
             * Domain knowledge (10%) 
             * Tools and methodologies (10%)
           - Experience level alignment(35% weight)
             * Years of relevant experience (15%)
             * Project complexity/scope (10%)
             * Industry relevance (10%)
           - Education and certification match (5% weight)
           - Soft skills and cultural fit indicators (5% weight)
        
        4. Provide your analysis in the following JSON format:
        {
            "match_score": <0-100 score>,
            "score_breakdown": {
                "skills_match": <0-100 score>,
                "experience_match": <0-100 score>,
                "education_match": <0-100 score>,
                "soft_skills_match": <0-100 score>
            },
            "key_matches": [<list of strongest matching skills/qualifications>],
            "missing_requirements": [<list of important requirements not found in resume>],
            "standout_qualifications": [<list of impressive qualifications beyond requirements>],
            "positive_comments": [<3-5 specific, detailed positive observations>],
            "negative_comments": [<2-3 specific improvement areas>],
            "summary": "<brief 2-3 sentence overall assessment>"
        }

        5.Analysis guidelines:
          - Look beyond exact keyword matches and recognize equivalent skills and technologies
          - Consider the recency of experiences (skills used in recent roles carry more weight)
          - Evaluate depth of experience, not just presence of keywords
          - Consider transferable skills from different industries when relevant
          - Recognize both stated and implied skills from project descriptions
          - Consider the specificity and detail level when candidates describe their experiences

        
        Be thorough in your analysis but concise in your explanations. Focus on substance over format in the candidate's materials.
        Understand that skills can be demonstrated through projects, work experience, or education - look for evidence across the entire resume.
        Account for both tech and non-tech roles appropriately.
        
        IMPORTANT: Your response must be valid JSON without any explanations before or after. No markdown formatting or code blocks.
        """
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Attempt to extract different sections from resume or job description
        
        Args:
            text: Resume or job description text
            
        Returns:
            Dictionary with sections
        """
        sections = {}
        
        # Common section headers in resumes and job descriptions
        section_patterns = [
            (r"(?:professional\s+)?(?:summary|profile|objective)", "summary"),
            (r"(?:work\s+|professional\s+)?(?:experience|history|employment)", "experience"),
            (r"(?:education|academic|qualification)", "education"),
            (r"(?:technical\s+)?skills(?:\s+and\s+abilities)?", "skills"),
            (r"(?:certification|accreditation)", "certifications"),
            (r"(?:project|portfolio)", "projects"),
            (r"(?:responsibility|duties|requirement)", "requirements"),
            (r"(?:qualification|what\s+you.*need)", "qualifications")
        ]
        
        # Find sections using regex
        section_matches = []
        for pattern, section_name in section_patterns:
            matches = list(re.finditer(r"(?i)(?:^|\n)(?:\d\.\s*|\•\s*)?(" + pattern + r")(?:\s*:|\.|\n)", text))
            for match in matches:
                section_matches.append((match.start(), match.group(), section_name))
        
        # Sort matches by position in text
        section_matches.sort(key=lambda x: x[0])
        
        # Extract text between sections
        for i in range(len(section_matches)):
            start_pos = section_matches[i][0]
            section_name = section_matches[i][2]
            
            # Determine end position (either next section or end of text)
            if i < len(section_matches) - 1:
                end_pos = section_matches[i+1][0]
            else:
                end_pos = len(text)
            
            # Extract section text
            section_text = text[start_pos:end_pos].strip()
            sections[section_name] = section_text
        
        return sections
    
    def _enhance_analysis_with_semantics(self, analysis: Dict[str, Any], job_description: str, resume: str) -> Dict[str, Any]:
        """
        Enhance Claude's analysis with additional semantic analysis
        
        Args:
            analysis: Initial analysis from Claude
            job_description: Job description text
            resume: Resume text
            
        Returns:
            Enhanced analysis
        """
        # Extract sections
        jd_sections = self._extract_sections(job_description)
        resume_sections = self._extract_sections(resume)
        
        # Preprocess texts
        jd_processed = self.semantic_analyzer.preprocess_text(job_description)
        resume_processed = self.semantic_analyzer.preprocess_text(resume)
        
        # Calculate semantic similarity scores
        tfidf_similarity = self.semantic_analyzer.calculate_tfidf_similarity(jd_processed, resume_processed)
        spacy_similarity = self.semantic_analyzer.calculate_spacy_similarity(jd_processed, resume_processed)
        
        # Extract skills
        jd_skills = self.semantic_analyzer.extract_skills(job_description)
        resume_skills = self.semantic_analyzer.extract_skills(resume)
        
        # Calculate skill overlap with semantic matching
        skill_overlap_score, matched_skills, missing_skills = self.semantic_analyzer.calculate_skill_overlap(jd_skills, resume_skills)
        
        # Update the analysis with semantic results
        semantic_analysis = {
            "semantic_similarity": {
                "tfidf_similarity": round(tfidf_similarity * 100, 2),
                "spacy_similarity": round(spacy_similarity * 100, 2),
                "skill_semantic_match": round(skill_overlap_score, 2)
            },
            "extracted_skills": {
                "jd_skills": jd_skills[:20],  # Limit to top 20 skills to avoid too much data
                "resume_skills": resume_skills[:20],
                "matched_skills": matched_skills[:20],
                "missing_skills": missing_skills[:20]
            }
        }
        
        # Blend Claude's analysis with semantic analysis
        if "missing_requirements" in analysis:
            # Add any missing skills that weren't already mentioned
            existing_missing = set([item.lower() for item in analysis["missing_requirements"]])
            for skill in missing_skills:
                if not any(skill.lower() in item.lower() for item in existing_missing):
                    analysis["missing_requirements"].append(skill)
        
        if "key_matches" in analysis:
            # Add any matched skills that weren't already mentioned
            existing_matches = set([item.lower() for item in analysis["key_matches"]])
            for skill in matched_skills:
                if not any(skill.lower() in item.lower() for item in existing_matches):
                    analysis["key_matches"].append(skill)
        
        # Add the semantic analysis section
        analysis["semantic_analysis"] = semantic_analysis
        
        # Recalculate skills match score as a blend of Claude's score and semantic score
        if "score_breakdown" in analysis and "skills_match" in analysis["score_breakdown"]:
            claude_skills_score = analysis["score_breakdown"]["skills_match"]
            blended_skills_score = (claude_skills_score * 0.7) + (skill_overlap_score * 0.3)
            analysis["score_breakdown"]["skills_match"] = round(blended_skills_score, 1)
            
            # Update overall match score
            if "match_score" in analysis:
                # Recalculate with original weightings
                weights = {
                    "skills_match": 0.45,
                    "experience_match": 0.3,
                    "education_match": 0.05,
                    "soft_skills_match": 0.1
                }
                
                weighted_sum = sum([
                    analysis["score_breakdown"].get(key, 0) * weight 
                    for key, weight in weights.items()
                ])
                
                analysis["match_score"] = round(weighted_sum, 1)
        
        return analysis
    
    def calculate_match(self, job_description: str, resume: str) -> Dict[str, Any]:
        """
        Calculate matching score between a job description and resume.
        
        Args:
            job_description: The job description text
            resume: The resume text
            
        Returns:
            Dict containing match score and detailed analysis
        """
        # Prepare the message content
        user_message = f"""
        # Job Description
        {job_description}
        
        # Resume
        {resume}
        
        Please analyze these documents and provide a matching score and detailed analysis in the requested JSON format.
        """
        
        # Call Claude API
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            )
            
            # Extract the JSON response
            try:
                # Get the response content
                content = response.content[0].text
                
                # Debug: Log the raw response
                st.write("Debug - Raw response length:", len(content))
                
                # Try different methods to extract JSON
                
                # Method 1: Look for JSON in code blocks
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.rfind("```")
                    if json_end > json_start:
                        json_str = content[json_start:json_end].strip()
                        
                # Method 2: Look for content that starts and ends with curly braces
                elif content.strip().startswith('{') and content.strip().endswith('}'):
                    json_str = content.strip()
                    
                # Method 3: Use regex to find JSON-like patterns
                else:
                    # Try to extract anything that looks like JSON
                    json_pattern = r'(\{[\s\S]*\})'
                    match = re.search(json_pattern, content)
                    if match:
                        json_str = match.group(1)
                    else:
                        json_str = content.strip()
                
                # Debug: Log the extracted JSON string
                if 'json_str' in locals():
                    st.write("Debug - Extracted JSON length:", len(json_str))
                    
                    # Try to parse JSON
                    try:
                        analysis = json.loads(json_str)
                        
                        # Enhance with semantic analysis
                        enhanced_analysis = self._enhance_analysis_with_semantics(analysis, job_description, resume)
                        
                        return enhanced_analysis
                    except json.JSONDecodeError as e:
                        st.error(f"JSON decode error: {e}")
                        # Try to fix common JSON issues
                        fixed_json = self._fix_json_format(json_str)
                        try:
                            analysis = json.loads(fixed_json)
                            
                            # Enhance with semantic analysis
                            enhanced_analysis = self._enhance_analysis_with_semantics(analysis, job_description, resume)
                            
                            return enhanced_analysis
                        except json.JSONDecodeError:
                            pass
                
                # If all methods fail, return error
                return {
                    "match_score": 0,
                    "error": "Failed to parse response",
                    "raw_response": content[:500] + "..." if len(content) > 500 else content
                }
                
            except Exception as e:
                st.error(f"Error parsing Claude's response: {e}")
                return {
                    "match_score": 0,
                    "error": f"Failed to parse response: {e}",
                    "raw_response": content[:500] + "..." if len(content) > 500 else content
                }
        except Exception as e:
            st.error(f"API Error: {e}")
            return {
                "match_score": 0,
                "error": f"API Error: {e}",
            }
    
    def _fix_json_format(self, json_str: str) -> str:
        """
        Try to fix common JSON formatting issues
        """
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix unquoted keys
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
        
        # Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        return json_str


# Helper functions to extract text from different file types
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_txt(txt_file):
    return txt_file.read().decode('utf-8')

def process_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            return extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            return extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            return extract_text_from_txt(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}. Please upload a PDF, DOCX, or TXT file.")
            return None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def create_radar_chart(categories, values):
    """Create a radar chart for the skills match visualization"""
    num_vars = len(categories)
    
    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Make the plot circular by appending the first value to the end
    values += values[:1]
    angles += angles[:1]
    categories += categories[:1]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories[:-1], size=12)
    
    # Draw the chart
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    # Add title
    plt.title('Skills Match Analysis', size=15, y=1.05)
    
    return fig
def main():
    st.set_page_config(
        page_title="JD Resume Matcher",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 0.5rem;
    }
    .match-score-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .score-label {
        font-size: 1.2rem;
        color: #555;
    }
    .high-score {
        color: #2E7D32;
        font-weight: bold;
    }
    .medium-score {
        color: #F9A825;
        font-weight: bold;
    }
    .low-score {
        color: #C62828;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        margin: 10px 0;
    }
    .section-title {
        background-color: #1E88E5;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>JD Resume Matcher</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-box'>Upload a job description and your resume to get an AI-powered analysis of how well your qualifications match the job requirements.</p>", unsafe_allow_html=True)
    
    # Sidebar for API key and options
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
        
        api_key = st.text_input("Claude API Key", type="password", 
                              help="Enter your Anthropic Claude API key. It will not be stored.")
        
        # Save API key to environment variable if provided
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        
        st.markdown("---")
        st.markdown("<h3>Analysis Options</h3>", unsafe_allow_html=True)
        
        show_semantic = st.checkbox("Show Semantic Analysis Details", value=True, 
                                  help="Show additional details from the semantic analysis")
        
        show_extracted_skills = st.checkbox("Show Extracted Skills", value=True,
                                         help="Show skills extracted from both documents")
        
        st.markdown("---")
        st.markdown("<h3>About</h3>", unsafe_allow_html=True)
        st.markdown("""
        This tool uses Claude API and semantic analysis to:
        - Compare your resume to job descriptions
        - Identify matching and missing skills
        - Calculate an overall match score
        - Provide suggestions for improvement
        """)
        
        st.markdown("---")
        st.markdown("© 2025 JD Resume Matcher")
    
    # Main content area divided into two columns
    col1, col2 = st.columns(2)
    
    # Left column for file uploads
    with col1:
        st.markdown("<h2 class='sub-header'>Upload Documents</h2>", unsafe_allow_html=True)
        
        # Job Description upload
        st.markdown("<h3>Job Description</h3>", unsafe_allow_html=True)
        jd_option = st.radio("Choose input method for job description:", 
                           ["Upload File", "Paste Text"], horizontal=True)
        
        job_description = None
        
        if jd_option == "Upload File":
            jd_file = st.file_uploader("Upload Job Description", 
                                     type=["pdf", "docx", "txt"],
                                     help="Upload a job description document")
            
            if jd_file is not None:
                job_description = process_uploaded_file(jd_file)
                if job_description:
                    st.success(f"Successfully processed {jd_file.name}")
                    with st.expander("Preview Job Description"):
                        st.text(job_description[:500] + "..." if len(job_description) > 500 else job_description)
        else:
            job_description = st.text_area("Paste Job Description", 
                                         height=300,
                                         help="Paste the job description text here")
        
        # Resume upload
        st.markdown("<h3>Resume</h3>", unsafe_allow_html=True)
        resume_option = st.radio("Choose input method for resume:", 
                               ["Upload File", "Paste Text"], horizontal=True)
        
        resume = None
        
        if resume_option == "Upload File":
            resume_file = st.file_uploader("Upload Resume", 
                                         type=["pdf", "docx", "txt"],
                                         help="Upload your resume")
            
            if resume_file is not None:
                resume = process_uploaded_file(resume_file)
                if resume:
                    st.success(f"Successfully processed {resume_file.name}")
                    with st.expander("Preview Resume"):
                        st.text(resume[:500] + "..." if len(resume) > 500 else resume)
        else:
            resume = st.text_area("Paste Resume", 
                               height=300,
                               help="Paste your resume text here")
        
        # Analysis button
        analyze_button = st.button("Analyze Match", type="primary", use_container_width=True)
        
        if analyze_button:
            if not api_key:
                st.error("Please enter your Claude API key in the sidebar")
            elif not job_description or not resume:
                st.error("Please provide both job description and resume")
            else:
                with st.spinner("Analyzing your resume against the job description..."):
                    # Initialize matcher with API key
                    matcher = JDResumeMatcher(api_key)
                    
                    # Calculate match
                    result = matcher.calculate_match(job_description, resume)
                    
                    # Store result in session state for the other column to use
                    st.session_state.analysis_result = result
                    st.session_state.job_description = job_description
                    st.session_state.resume = resume
                    
                    # Rerun to update the right column
                    st.rerun()
    
    # Right column for results
    with col2:
        st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
        
        # Check if results are available
        if 'analysis_result' not in st.session_state:
            st.info("Upload documents and click 'Analyze Match' to see results here")
        else:
            result = st.session_state.analysis_result
            
            # Display match score
            score = result.get("match_score", 0)
            st.markdown("<div class='match-score-container'>", unsafe_allow_html=True)
            st.markdown("<p class='score-label'>Overall Match Score</p>", unsafe_allow_html=True)
            
            # Determine score class based on value
            score_class = "high-score" if score >= 75 else "medium-score" if score >= 50 else "low-score"
            st.markdown(f"<h1 class='{score_class}'>{score}%</h1>", unsafe_allow_html=True)
            
            # Score interpretation
            if score >= 75:
                message = "Great match! You're well qualified for this position."
            elif score >= 60:
                message = "Good match. Consider highlighting your relevant experience more clearly."
            elif score >= 40:
                message = "Moderate match. You may need to acquire some additional skills or emphasize relevant experience."
            else:
                message = "Low match. This position may require qualifications you don't currently have."
            
            st.markdown(f"<p>{message}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Score breakdown
            if "score_breakdown" in result:
                st.markdown("<h3>Score Breakdown</h3>", unsafe_allow_html=True)
                
                # Get scores
                breakdown = result["score_breakdown"]
                categories = []
                values = []
                
                if "skills_match" in breakdown:
                    categories.append("Skills")
                    values.append(breakdown["skills_match"])
                
                if "experience_match" in breakdown:
                    categories.append("Experience")
                    values.append(breakdown["experience_match"])
                
                if "education_match" in breakdown:
                    categories.append("Education")
                    values.append(breakdown["education_match"])
                
                if "soft_skills_match" in breakdown:
                    categories.append("Soft Skills")
                    values.append(breakdown["soft_skills_match"])
                
                # Create radar chart
                if len(categories) > 0:
                    fig = create_radar_chart(categories, values)
                    st.pyplot(fig)
                
                # Also display as a table
                breakdown_df = pd.DataFrame({
                    "Category": categories,
                    "Score": values
                })
                st.dataframe(breakdown_df, hide_index=True, use_container_width=True)
            
            # Key matches
            if "key_matches" in result and result["key_matches"]:
                st.markdown("<h3 class='section-title'>Key Matches</h3>", unsafe_allow_html=True)
                for item in result["key_matches"]:
                    st.markdown(f"✅ {item}")
            
            # Missing requirements
            if "missing_requirements" in result and result["missing_requirements"]:
                st.markdown("<h3 class='section-title'>Missing Requirements</h3>", unsafe_allow_html=True)
                for item in result["missing_requirements"]:
                    st.markdown(f"❌ {item}")
            
            # Standout qualifications
            if "standout_qualifications" in result and result["standout_qualifications"]:
                st.markdown("<h3 class='section-title'>Standout Qualifications</h3>", unsafe_allow_html=True)
                for item in result["standout_qualifications"]:
                    st.markdown(f"⭐ {item}")
            
            # Positive comments
            if "positive_comments" in result and result["positive_comments"]:
                st.markdown("<h3 class='section-title'>Positive Feedback</h3>", unsafe_allow_html=True)
                for item in result["positive_comments"]:
                    st.markdown(f"👍 {item}")
            
            # Negative comments
            if "negative_comments" in result and result["negative_comments"]:
                st.markdown("<h3 class='section-title'>Areas for Improvement</h3>", unsafe_allow_html=True)
                for item in result["negative_comments"]:
                    st.markdown(f"💡 {item}")
            
            # Semantic analysis details
            if show_semantic and "semantic_analysis" in result:
                st.markdown("<h3 class='section-title'>Semantic Analysis</h3>", unsafe_allow_html=True)
                
                semantic = result["semantic_analysis"]
                
                if "semantic_similarity" in semantic:
                    sim = semantic["semantic_similarity"]
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("TF-IDF Similarity", f"{sim.get('tfidf_similarity', 0):.1f}%")
                    
                    with col_b:
                        st.metric("spaCy Similarity", f"{sim.get('spacy_similarity', 0):.1f}%")
                    
                    with col_c:
                        st.metric("Skill Semantic Match", f"{sim.get('skill_semantic_match', 0):.1f}%")
            
            # Extracted skills
            if show_extracted_skills and "semantic_analysis" in result and "extracted_skills" in result["semantic_analysis"]:
                st.markdown("<h3 class='section-title'>Extracted Skills</h3>", unsafe_allow_html=True)
                
                skills = result["semantic_analysis"]["extracted_skills"]
                
                # Create tabs for different skill categories
                tabs = st.tabs(["JD Skills", "Resume Skills", "Matched Skills", "Missing Skills"])
                
                with tabs[0]:
                    if "jd_skills" in skills and skills["jd_skills"]:
                        st.write(", ".join(skills["jd_skills"]))
                    else:
                        st.info("No skills extracted from job description")
                
                with tabs[1]:
                    if "resume_skills" in skills and skills["resume_skills"]:
                        st.write(", ".join(skills["resume_skills"]))
                    else:
                        st.info("No skills extracted from resume")
                
                with tabs[2]:
                    if "matched_skills" in skills and skills["matched_skills"]:
                        st.write(", ".join(skills["matched_skills"]))
                    else:
                        st.info("No matching skills found")
                
                with tabs[3]:
                    if "missing_skills" in skills and skills["missing_skills"]:
                        st.write(", ".join(skills["missing_skills"]))
                    else:
                        st.info("No missing skills identified")
            
            # Summary
            if "summary" in result and result["summary"]:
                st.markdown("<h3 class='section-title'>Summary</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='info-box'>{result['summary']}</div>", unsafe_allow_html=True)
                
                # Add download button for the full analysis
                st.download_button(
                    label="Download Full Analysis (JSON)",
                    data=json.dumps(result, indent=2),
                    file_name="resume_analysis.json",
                    mime="application/json"
                )
    
    # Bottom section for data visualization (full width)
    if 'analysis_result' in st.session_state:
        with st.expander("Advanced Visualizations", expanded=False):
            st.markdown("<h3>Skill Distribution</h3>", unsafe_allow_html=True)
            
            # Check if we have the required data
            if ("semantic_analysis" in st.session_state.analysis_result and 
                "extracted_skills" in st.session_state.analysis_result["semantic_analysis"]):
                
                skills_data = st.session_state.analysis_result["semantic_analysis"]["extracted_skills"]
                
                # Create a DataFrame of skills
                jd_skills = pd.DataFrame({"Skill": skills_data.get("jd_skills", []), 
                                        "Source": ["Job Description"] * len(skills_data.get("jd_skills", []))})
                
                resume_skills = pd.DataFrame({"Skill": skills_data.get("resume_skills", []), 
                                           "Source": ["Resume"] * len(skills_data.get("resume_skills", []))})
                
                all_skills = pd.concat([jd_skills, resume_skills])
                
                if not all_skills.empty:
                    # Count skills by source
                    skill_counts = all_skills.groupby("Source").count().reset_index()
                    
                    # Plot bar chart
                    st.bar_chart(skill_counts.set_index("Source"))
                    
                    # Create word frequency count
                    skill_freq = all_skills["Skill"].value_counts().reset_index()
                    skill_freq.columns = ["Skill", "Count"]
                    
                    # Display most common skills
                    st.markdown("<h4>Most Common Skills</h4>", unsafe_allow_html=True)
                    st.dataframe(skill_freq.head(10), hide_index=True)
                else:
                    st.info("No skills data available for visualization")
            else:
                st.info("Skill distribution data not available")


if __name__ == "__main__":
    main()