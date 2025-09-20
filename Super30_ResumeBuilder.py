import streamlit as st
import os
import json
import fitz  # PyMuPDF
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- Constants & Page Config ---
FONT_FAMILY = "Helvetica"
st.set_page_config(
    page_title="AI Resume Co-pilot",
    page_icon="📄",
    layout="wide"
)

# --- Helper function to safely encode text ---
def clean_text(text):
    """
    Sanitizes text for FPDF by replacing special Unicode characters with ASCII
    equivalents and handling encoding. This is a more robust version.
    """
    text = str(text).strip()
    replacements = {
        '\u00A0': ' ',  # Non-breaking space
        '\r': '',
        '\n': ' ',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '–': '-',  # En dash
        '—': '-',  # Em dash
        '‑': '-',  # Non-breaking hyphen
        '•': '*'   # Bullet
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Encode to latin-1, replacing any remaining unsupported characters
    return text.encode('latin-1', 'replace').decode('latin-1')

# --- PDF Generation Class ---

class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        pass

    def write_resume_from_markdown(self, markdown_text):
        """Parses a markdown-like text and formats it into the PDF."""
        lines = markdown_text.split('\n')
        
        if lines:
            name = lines.pop(0).replace('#', '').strip()
            self.set_font(FONT_FAMILY, 'B', 24)
            self.cell(0, 10, clean_text(name), 0, 1, 'C')

        if lines:
            contact = lines.pop(0).strip()
            self.set_font(FONT_FAMILY, '', 10)
            self.cell(0, 10, clean_text(contact), 0, 1, 'C')
            self.ln(5)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('## '):
                self.ln(4)
                self.set_font(FONT_FAMILY, 'B', 14)
                self.cell(0, 10, clean_text(line.replace('##', '').strip()).upper(), 0, 1, 'L')
                self.ln(2)
            elif line.startswith('**'):
                self.set_font(FONT_FAMILY, 'B', 11)
                self.multi_cell(0, 5, clean_text(line.replace('**', '')))
            elif line.startswith('* '):
                self.set_font(FONT_FAMILY, '', 11)
                self.multi_cell(0, 5, clean_text(line))
            else:
                self.set_font(FONT_FAMILY, 'I', 10)
                self.multi_cell(0, 5, clean_text(line))

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
         The JSON object must have the following keys: 'name', 'contact' (as a dictionary with 'email', 'phone', 'linkedin_url'), 'summary', 'experience' (as a list of objects, each with 'title', 'company', 'duration', and 'description'), 'education' (a list of objects with 'institution', 'degree', 'duration'), and 'skills' (a list of strings).
         Pay close attention to the 'Education' and 'Skills' sections to ensure they are parsed correctly and not mixed.
         - For 'education': An entry should only be included if it is clearly an educational institution, degree, or certification. For entries like fellowships or non-degree programs, parse the main line as 'degree' and the organization as 'institution'. If duration is present, extract it.
         - For 'skills': Extract only the specific skills, which are often listed under a 'Skills' heading or similar. Parse a comma-separated list into a JSON list of individual skill strings.
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

@st.cache_data(show_spinner="Formatting your resume for download...")
def generate_markdown_resume(_llm, resume_data):
    """Uses LLM to convert resume JSON into a clean Markdown format."""
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional resume typesetter. Your task is to convert the following resume data from JSON format into a clean, well-structured Markdown document.

Follow these formatting rules STRICTLY:
- The very first line MUST be the candidate's name, preceded by '# '.
- The second line MUST be the contact information, formatted as a single line (e.g., 'email | phone | linkedin').
- Section titles (like 'Professional Summary', 'Work Experience') MUST be preceded by '## '.
- For each experience entry, the job title and company MUST be on one line, in bold (e.g., '**Senior Developer at Tech Corp**').
- The duration for each job or education MUST be on its own line, immediately following the title line.
- Each point in a job description MUST start with '* '.
- For education, the institution MUST be in bold (e.g., '**State University**').
- The degree and duration for education should be on the next line.
- The skills should be a single comma-separated line.
- Do not add any extra text, comments, or explanations. Only output the Markdown resume."""),
        ("user", "Here is the resume data:\n\n{resume_json}"),
    ])
    
    chain = prompt | _llm | parser
    response = chain.invoke({"resume_json": json.dumps(resume_data)})
    return response

# --- Streamlit UI ---

st.title("📄 AI Resume Co-pilot")
st.write("Generate a professional resume from your LinkedIn profile or existing resume and tailor it to a specific job in seconds.")

# --- Session State Initialization ---
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'markdown_resume' not in st.session_state:
    st.session_state.markdown_resume = None

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
    st.session_state.markdown_resume = None # Clear old markdown on new generation
    if not groq_api_key:
        st.error("Please enter your GROQ API key.")
    elif not uploaded_pdf:
        st.error("Please upload your profile or resume PDF.")
    else:
        llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        
        document_text = extract_text_from_pdf(uploaded_pdf)
        
        if document_text:
            resume_json = generate_resume_from_text(llm, document_text)
            if job_description.strip():
                final_resume_json = customize_resume_for_job(llm, resume_json, job_description)
            else:
                final_resume_json = resume_json
            
            st.session_state.resume_data = final_resume_json
            st.success("Resume generated successfully! You can now edit the content below.")

if st.session_state.resume_data:
    st.header("📝 Edit Your Resume")
    
    resume = st.session_state.resume_data
    
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

    skills_list = resume.get('skills', [])
    if isinstance(skills_list, list):
        skills_str = ", ".join(skills_list)
    else:
        skills_str = skills_list
    resume['skills'] = st.text_area("Skills (comma-separated)", skills_str, key='skills')

    # --- PDF Download Button ---
    st.header("📥 Download Your Resume")
    
    if st.button("Prepare PDF for Download"):
        if not groq_api_key:
            st.error("Please enter your GROQ API key in the sidebar to format and download.")
        else:
            llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
            
            # Convert skills back to a list if it's a string
            if isinstance(resume['skills'], str):
                resume['skills'] = [skill.strip() for skill in resume['skills'].split(',')]

            markdown_resume = generate_markdown_resume(llm, resume)
            st.session_state.markdown_resume = markdown_resume

    if st.session_state.markdown_resume:
        st.subheader("Formatted Resume Preview")
        st.markdown(st.session_state.markdown_resume)

        pdf = PDF()
        pdf.add_page()
        pdf.write_resume_from_markdown(st.session_state.markdown_resume)
        
        pdf_output = pdf.output(dest='S').encode('latin-1')

        st.download_button(
            label="✅ Click to Download PDF",
            data=pdf_output,
            file_name=f"{st.session_state.resume_data.get('name', 'resume').replace(' ', '_').lower()}_resume.pdf",
            mime="application/pdf"
        )
