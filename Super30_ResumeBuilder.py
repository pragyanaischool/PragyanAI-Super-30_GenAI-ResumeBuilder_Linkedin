import streamlit as st
import os
import json
import fitz  # PyMuPDF
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- Constants & Page Config ---
st.set_page_config(
    page_title="PragyanAI - AI Resume Co-pilot",
    page_icon="📄",
    layout="wide"
)

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

@st.cache_data(show_spinner="Rewriting your resume for maximum impact...")
def rewrite_resume_for_impact(_llm, resume_data):
    """Uses LLM to rewrite the resume content to be more effective."""
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a world-class career coach and resume writer. You will be given a resume in JSON format. Your task is to rewrite the 'summary' and the 'description' for each 'experience' entry to be more impactful, professional, and achievement-oriented.
         - For the summary: Craft a dynamic and compelling professional summary that immediately grabs the reader's attention and highlights the candidate's unique value proposition.
         - For experience descriptions: Transform the descriptions from a list of duties into a showcase of accomplishments. Use the STAR (Situation, Task, Action, Result) method where possible. Start each bullet point with a strong action verb. Quantify results with numbers, percentages, or other metrics whenever you can infer them or suggest placeholders (e.g., 'Increased efficiency by over 25%'). Ensure all bullet points still start with '*'.
         - Do not change any other part of the resume JSON. Return the entire modified JSON object."""),
        ("user", "Here is the resume to rewrite:\n\n{resume}"),
        ("system", "Please provide the rewritten resume in JSON format only.")
    ])
    
    chain = prompt | _llm | parser
    response = chain.invoke({"resume": json.dumps(resume_data)})
    return response

@st.cache_data(show_spinner="Generating a draft cover letter...")
def generate_cover_letter(_llm, resume_data, job_description):
    """Uses LLM to generate a cover letter."""
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional career writer. Your task is to write a compelling and professional cover letter based on a candidate's resume and a specific job description.
        The cover letter should be well-structured, concise, and tailored to the job.
        - **Introduction:** Start with a strong opening that states the position being applied for and where it was seen.
        - **Body Paragraphs:** Create 2-3 paragraphs that connect the candidate's key experiences and skills from their resume to the specific requirements listed in the job description. Do not just repeat the resume; explain *how* their experience is relevant.
        - **Conclusion:** End with a strong closing paragraph that reiterates interest in the role and includes a clear call to action (e.g., expressing eagerness to discuss their qualifications in an interview).
        - **Formatting:** Use professional and friendly language. The output should be a single block of text formatted in Markdown.
        """),
        ("user", "Here is the candidate's resume:\n\n{resume}\n\nHere is the job description they are applying for:\n\n{job_post}"),
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
if 'markdown_resume' not in st.session_state:
    st.session_state.markdown_resume = None
if 'cover_letter' not in st.session_state:
    st.session_state.cover_letter = None

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
    st.session_state.cover_letter = None  # Clear old cover letter
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

    # --- AI Writing Assistants ---
    st.header("✨ AI Writing Assistants")
    st.write("Use AI to further enhance your resume and create a cover letter.")
    
    assist_col1, assist_col2 = st.columns(2)
    with assist_col1:
        if st.button("🚀 Rewrite Resume for Impact"):
            if not groq_api_key:
                st.error("Please enter your GROQ API key in the sidebar.")
            else:
                llm = ChatGroq(temperature=0.4, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
                if isinstance(st.session_state.resume_data['skills'], str):
                    st.session_state.resume_data['skills'] = [skill.strip() for skill in st.session_state.resume_data['skills'].split(',')]
                
                rewritten_resume_json = rewrite_resume_for_impact(llm, st.session_state.resume_data)
                st.session_state.resume_data = rewritten_resume_json
                st.success("Resume rewritten! The fields above have been updated with the new content.")
                st.rerun()

    with assist_col2:
        if st.button("✉️ Generate Cover Letter"):
            if not groq_api_key:
                st.error("Please enter your GROQ API key in the sidebar.")
            elif not job_description.strip():
                st.error("Please provide a job description in the sidebar to generate a cover letter.")
            else:
                llm = ChatGroq(temperature=0.5, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
                if isinstance(st.session_state.resume_data['skills'], str):
                    st.session_state.resume_data['skills'] = [skill.strip() for skill in st.session_state.resume_data['skills'].split(',')]

                cover_letter_text = generate_cover_letter(llm, st.session_state.resume_data, job_description)
                st.session_state.cover_letter = cover_letter_text
    
    if st.session_state.cover_letter:
        st.subheader("Generated Cover Letter")
        st.session_state.cover_letter = st.text_area("Edit your cover letter:", value=st.session_state.cover_letter, height=400)

    # --- Markdown Download Buttons ---
    st.header("📥 Download Your Documents")
    
    if st.button("Prepare Documents for Download"):
        if not groq_api_key:
            st.error("Please enter your GROQ API key in the sidebar to format and download.")
        else:
            llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
            
            if isinstance(resume['skills'], str):
                resume['skills'] = [skill.strip() for skill in resume['skills'].split(',')]

            markdown_resume = generate_markdown_resume(llm, resume)
            st.session_state.markdown_resume = markdown_resume

    if st.session_state.markdown_resume:
        st.subheader("Formatted Resume Preview")
        st.markdown(st.session_state.markdown_resume)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📝 Download Resume as Markdown",
                data=st.session_state.markdown_resume,
                file_name=f"{st.session_state.resume_data.get('name', 'resume').replace(' ', '_').lower()}_resume.md",
                mime="text/markdown"
            )
        if st.session_state.cover_letter:
            with dl_col2:
                st.download_button(
                    label="✉️ Download Cover Letter",
                    data=st.session_state.cover_letter,
                    file_name=f"{st.session_state.resume_data.get('name', 'resume').replace(' ', '_').lower()}_cover_letter.md",
                    mime="text/markdown"
                )
