# ATS-TRACKER
A powerful web application that leverages AI and semantic analysis to match resumes with job descriptions, helping job seekers optimize their applications and recruiters find the best candidates.
JD Resume Matcher
Show Image
Show Image
Show Image
A powerful web application that leverages AI and semantic analysis to match resumes with job descriptions, helping job seekers optimize their applications and recruiters find the best candidates.
Show Image
🚀 Features

AI-Powered Analysis: Uses Anthropic's Claude API to perform deep comparison between job descriptions and resumes
Semantic Matching: Goes beyond keyword matching with NLP-based semantic analysis
Skills Extraction: Automatically identifies skills in both documents
Detailed Scoring: Provides overall match score and breakdowns across multiple categories
Visualization: Visual representation of match scores with radar charts
File Support: Upload PDFs, Word documents, or paste text directly
Actionable Feedback: Identifies strengths, gaps, and suggested improvements

📋 Requirements

Python 3.8+
Anthropic API key (Get one here)
Required Python packages (see requirements.txt)

🛠️ Installation

Clone this repository:
bashgit clone https://github.com/yourusername/jd-resume-matcher.git
cd jd-resume-matcher

Install required packages:
bashpip install -r requirements.txt

(Optional) Download required NLP models:
bashpython -m spacy download en_core_web_md
python -m nltk.downloader punkt stopwords


🖥️ Usage

Run the Streamlit app:
bashstreamlit run app.py

Open your web browser and navigate to http://localhost:8501
Enter your Anthropic Claude API key in the sidebar
Upload or paste your job description and resume
Click "Analyze Match" to get results

🧩 How It Works
The Analysis Process

Text Extraction: Extracts text from uploaded documents (PDF, DOCX, TXT)
Claude AI Analysis: Uses Claude to identify key requirements, skills, and qualifications
Semantic Analysis: Applies NLP techniques to find semantic similarities beyond exact keyword matches
Skills Extraction: Uses spaCy NER and pattern matching to identify technical skills and competencies
Score Calculation: Computes match scores based on multiple weighted factors
Visualization: Presents results in an intuitive interface with helpful visualizations

Scoring Methodology
The overall match score is calculated based on these weighted components:

Skills Match (55%)

Technical skills (30%)
Domain knowledge (10%)
Tools and methodologies (10%)


Experience Match (35%)

Relevant experience years (15%)
Project complexity/scope (10%)
Industry relevance (10%)


Education/Certification Match (5%)
Soft Skills Match (5%)

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgements

Anthropic Claude API for AI-powered analysis
Streamlit for the web interface
spaCy for NLP capabilities
NLTK for natural language processing
PyPDF2 for PDF processing
python-docx for DOCX processing
scikit-learn for TF-IDF and cosine similarity
