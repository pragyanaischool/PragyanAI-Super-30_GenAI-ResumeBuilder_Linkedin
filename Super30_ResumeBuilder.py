import streamlit as st
import os
import json
import fitz  # PyMuPDF
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Constants & Page Config ---
FONT_FAMILY = "Helvetica"
st.set_page_config(
    page_title="PragyanAI - AI Resume Co-pilot",
    page_icon="📄",
    layout="wide"
)

# --- PDF Generation Class ---

class PDF(FPDF):
    def header(self):
        pass # No header

    def footer(self):
        pass # No footer
        
    def add_section_title(self, title):
        self.set_font(FONT_FAMILY, 'B', 14)
        safe_title = title.encode('latin-1', 'replace').decode('latin-1')
        self.cell(0, 10, safe_title.upper(), 0, 1, 'L')
        self.ln(2) # Add a little space after the title

    def add_body_text(self, text):
        self.set_font(FONT_FAMILY, '', 11)
        safe_text = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 5, safe_text)
        self.ln(4)

    def add_experience(self, exp_list):
        for exp in exp_list:
            self.set_font(FONT_FAMILY, 'B', 11)
            title_company = f"{exp.get('title', '')} at {exp.get('company', '')}"
            safe_title_company = title_company.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, safe_title_company)

            self.set_font(FONT_FAMILY, 'I', 10)
            duration = f"{exp.get('duration', '')}"
            safe_duration = duration.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, safe_duration)
            
            self.set_font(FONT_FAMILY, '', 11)
            # Handle bullet points in description, replacing special characters
            description = exp.get('description', '').replace('•', '*')
            for point in description.split('*'):
                point = point.strip()
                if point:
                    safe_point = f"* {point}".encode('latin-1', 'replace').decode('latin-1')
                    self.multi_cell(0, 5, safe_point)
            self.ln(3)

    def add_education(self, edu_list):
        for edu in edu_list:
            self.set_font(FONT_FAMILY, 'B', 11)
            institution = edu.get('institution', '')
            safe_institution = institution.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, safe_institution)

            self.set_font(FONT_FAMILY, '', 11)
            degree_duration = f"{edu.get('degree', '')} ({edu.get('duration', '')})"
            safe_degree_duration = degree_duration.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, safe_degree_duration)
            self.ln(3)

# --- Core Functions ---

def extract_text_from_pdf(pdf_file):
    """Extracts text content from an uploaded PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

@st.cache_data(show_spinner="Generating resume from your profile...")
def generate_resume_from_text(_llm, text):
    """Uses LLM to parse text and generate a structured resume in JSON."""
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert resume writer. Your task is to analyze the provided text from a LinkedIn profile or an existing resume and extract the user's information into a structured JSON format. 
         The JSON object should have the following keys: 'name', 'contact' (as a dictionary with 'email', 'phone', 'linkedin_url'), 'summary', 'experience' (as a list of objects, each with 'title', 'company', 'duration', and 'description'), 'education' (a list of objects with 'institution', 'degree', 'duration'), and 'skills' (a list of strings).
         Clean and format the text professionally. For job descriptions, convert paragraphs into concise bullet points, each starting with '*'. Infer missing details logically if necessary, but don't invent information that isn't suggested by the text."""),
        ("user", "Here is the text from the document:\n\n{document_text}"),
        ("system", "Please provide the output in JSON format only.")
    ])
    
    chain = prompt | _llm | parser
    response = chain.invoke({"document_text": text})
    return response

@st.cache_data(show_spinner="Customizing resume for the job post...")
def customize_resume_for_job(_llm, resume_data, job_description):
    """Uses LLM to tailor the resume for a specific job description."""
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI-powered resume co-pilot. You will be given a resume in JSON format and a job description. 
         Your task is to analyze the job description and strategically rewrite the 'summary' and the 'description' for each 'experience' entry in the resume. 
         The goal is to align the candidate's skills and experience with the requirements of the job.
         - For the summary: Create a powerful, concise professional summary that highlights the candidate's most relevant qualifications for this specific role.
         - For experience descriptions: Rephrase the bullet points to use keywords and action verbs from the job description. Quantify achievements where possible and emphasize results that match the employer's needs. Ensure all bullet points start with '*'.
         - Do not change any other part of the resume JSON. Return the entire modified JSON object."""),
        ("user", "Here is the current resume:\n\n{resume}\n\nHere is the job description:\n\n{job_post}"),
        ("system", "Please provide the updated resume in JSON format only.")
    ])
    
    chain = prompt | _llm | parser
    response = chain.invoke({"resume": json.dumps(resume_data), "job_post": job_description})
    return response

# --- Streamlit UI ---

st.title("📄 PragyanAI - AI Resume Co-pilot")
st.write("Generate a professional resume from your LinkedIn profile or existing resume and tailor it to a specific job in seconds.")

# --- Session State Initialization ---
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    
    groq_api_key = st.text_input("Enter your GROQ API Key", type="password")
    
    st.subheader("1. Provide Your Profile")
    input_method = st.radio(
        "Choose your input method:",
        ("Upload LinkedIn PDF", "Upload Resume PDF"),
        key="input_method"
    )

    uploaded_pdf = None
    if input_method == "Upload LinkedIn PDF":
        st.info("Go to your LinkedIn profile > More > Save to PDF. Then upload the file below.")
        uploaded_pdf = st.file_uploader("Upload your LinkedIn PDF", type="pdf", key="linkedin_pdf")
    else:
        st.info("Upload your existing resume in PDF format.")
        uploaded_pdf = st.file_uploader("Upload your Resume PDF", type="pdf", key="resume_pdf")

    st.subheader("2. Add Job Description")
    st.info("Paste a job description below to tailor your resume for that specific role.")
    job_description = st.text_area("Paste the job description here", height=200)

    generate_button = st.button("✨ Generate & Customize Resume")

# --- Main Content Area ---
if generate_button:
    if not groq_api_key:
        st.error("Please enter your GROQ API key.")
    elif not uploaded_pdf:
        st.error("Please upload your profile or resume PDF.")
    else:
        llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
        
        # 1. Extract text from PDF
        document_text = extract_text_from_pdf(uploaded_pdf)
        
        if document_text:
            # 2. Generate base resume
            resume_json = generate_resume_from_text(llm, document_text)
            
            # 3. Customize if job description is provided
            if job_description.strip():
                final_resume_json = customize_resume_for_job(llm, resume_json, job_description)
            else:
                final_resume_json = resume_json
            
            st.session_state.resume_data = final_resume_json
            st.success("Resume generated successfully! You can now edit the content below.")

if st.session_state.resume_data:
    st.header("📝 Edit Your Resume")
    
    resume = st.session_state.resume_data
    
    # --- Editable Fields ---
    # We use keys to ensure widgets are re-rendered correctly
    resume['name'] = st.text_input("Name", resume.get('name', ''), key='name')
    
    contact_info = resume.get('contact', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        contact_info['email'] = st.text_input("Email", contact_info.get('email', ''), key='email')
    with col2:
        contact_info['phone'] = st.text_input("Phone", contact_info.get('phone', ''), key='phone')
    with col3:
        contact_info['linkedin_url'] = st.text_input("LinkedIn URL", contact_info.get('linkedin_url', ''), key='linkedin_url')
    resume['contact'] = contact_info

    resume['summary'] = st.text_area("Professional Summary", resume.get('summary', ''), height=150, key='summary')

    st.subheader("Work Experience")
    for i, exp in enumerate(resume.get('experience', [])):
        with st.expander(f"{exp.get('title', 'Job Title')} at {exp.get('company', 'Company')}", expanded=True):
            exp['title'] = st.text_input("Title", exp.get('title', ''), key=f"exp_title_{i}")
            exp['company'] = st.text_input("Company", exp.get('company', ''), key=f"exp_company_{i}")
            exp['duration'] = st.text_input("Duration", exp.get('duration', ''), key=f"exp_duration_{i}")
            exp['description'] = st.text_area("Description", exp.get('description', ''), height=150, key=f"exp_desc_{i}")

    st.subheader("Education")
    for i, edu in enumerate(resume.get('education', [])):
         with st.expander(f"{edu.get('institution', 'Institution')}", expanded=True):
            edu['institution'] = st.text_input("Institution", edu.get('institution', ''), key=f"edu_inst_{i}")
            edu['degree'] = st.text_input("Degree/Field of Study", edu.get('degree', ''), key=f"edu_degree_{i}")
            edu['duration'] = st.text_input("Duration", edu.get('duration', ''), key=f"edu_duration_{i}")

    resume['skills'] = st.text_area("Skills (comma-separated)", ", ".join(resume.get('skills', [])), key='skills')

    # --- PDF Download Button ---
    st.header("📥 Download Your Resume")
    
    pdf = PDF()
    pdf.add_page()
    
    # Header section
    pdf.set_font(FONT_FAMILY, 'B', 24)
    safe_name = resume.get('name', 'Your Name').encode('latin-1', 'replace').decode('latin-1')
    pdf.cell(0, 10, safe_name, 0, 1, 'C')

    pdf.set_font(FONT_FAMILY, '', 10)
    contact_str = f"{resume['contact'].get('email', '')} | {resume['contact'].get('phone', '')} | {resume['contact'].get('linkedin_url', '')}"
    safe_contact_str = contact_str.encode('latin-1', 'replace').decode('latin-1')
    pdf.cell(0, 10, safe_contact_str, 0, 1, 'C')
    pdf.ln(5)

    # Summary
    pdf.add_section_title('Professional Summary')
    pdf.add_body_text(resume.get('summary', ''))

    # Experience
    pdf.add_section_title('Work Experience')
    pdf.add_experience(resume.get('experience', []))

    # Education
    pdf.add_section_title('Education')
    pdf.add_education(resume.get('education', []))

    # Skills
    pdf.add_section_title('Skills')
    pdf.add_body_text(", ".join(resume.get('skills', []) if isinstance(resume.get('skills'), list) else resume.get('skills').split(', ')))

    pdf_output = pdf.output(dest='S').encode('latin-1')

    st.download_button(
        label="Download Resume as PDF",
        data=pdf_output,
        file_name=f"{resume.get('name', 'resume').replace(' ', '_').lower()}_resume.pdf",
        mime="application/pdf"
    )
