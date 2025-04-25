import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("🧠 AI Resume Analyzer")

# --- Extract text from PDF resume ---
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# --- Match score using TF-IDF + cosine similarity ---
def calculate_match_score(resume_text, jd_text):
    docs = [resume_text, jd_text]
    tfidf = TfidfVectorizer().fit_transform(docs)
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

# --- AI Feedback using GPT-3.5-Turbo only ---
def gpt_feedback(resume_text, jd_text):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""You are an AI resume reviewer. Analyze the resume and job description below, and suggest:
1. Missing skills
2. Role mismatch (if any)
3. Tips to improve the resume for better fit

Resume:
{resume_text[:1000]}...

Job Description:
{jd_text[:1000]}...
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # ✅ Uses only the supported model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ GPT error: {e}"

# --- Streamlit UI ---
resume = st.file_uploader("📄 Upload your resume (PDF only)", type="pdf")
job_desc = st.text_area("📋 Paste the Job Description here")
use_gpt = st.checkbox("Also generate AI-based feedback (requires OpenAI API key)")

if st.button("Analyze"):
    if resume and job_desc:
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(resume)
            score = calculate_match_score(resume_text, job_desc)
            st.success(f"✅ Resume Match Score: {score}%")

            if use_gpt:
                st.markdown("### 💬 GPT Feedback")
                feedback = gpt_feedback(resume_text, job_desc)
                st.info(feedback)
    else:
        st.warning("Please upload a resume and paste a job description.")
