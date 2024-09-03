import streamlit as st
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import PyPDF2
import os
from dotenv import load_dotenv 

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Extract text from PDF
def extract_text_from_pdf(file_like):
    pdfreader = PyPDF2.PdfReader(file_like)
    raw_text = ''
    for page in pdfreader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

openai.api_key = os.getenv("OPENAI_API_KEY")
# Define your functions to generate questions
def mcqs(num_questions, difficulty_level, context):
    prompt = f"""Generate {num_questions} {difficulty_level} multiple-choice questions based on the following text. Format each question as follows:

    Question: <Question Text>
    a) <Option 1>
    b) <Option 2>
    c) <Option 3>
    d) <Option 4>

    Answer: <Correct Answer>

    Text for generating questions:
    {context}
"""
    llm = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.5
    )
    return llm.choices[0]['message']['content'].strip()

def fill_in_the_blanks(num_questions, difficulty_level, context):
    prompt = f"""Generate {num_questions} fill-in-the-blank questions based on the following text. Format each question as follows:

    Question: <Question with a blank>
    Answer: <Correct Answer>

    Text for generating questions:
    {context}
"""
    llm = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.5
    )
    return llm.choices[0]['message']['content'].strip()

def true_or_false(num_questions, difficulty_level, context):
    prompt = f"""Generate {num_questions} true/false questions based on the following text. Format each question as follows:

    Question: <Statement>
    Answer: <True or False>

    Text for generating questions:
    {context}
"""
    llm = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.5
    )
    return llm.choices[0]['message']['content'].strip()

def text_based(num_questions, difficulty_level, context):
    prompt = f"""Generate {num_questions} questions and answers based on the following text. Format each question and answer pair as follows:

    Question: <Question based on the text>
    Answer: <Correct Answer based on the text>

    Text for generating questions:
    {context}
"""
    llm = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.5
    )
    return llm.choices[0]['message']['content'].strip()

# Streamlit app UI
st.title("Question Answering Generator App")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)


    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    document_search = FAISS.from_texts(texts, embeddings)
    query = "All concepts from the chapter"
    result = document_search.similarity_search(query)

    # Inputs
    num_questions = st.number_input("Enter the number of questions:", min_value=1, value=30)
    difficulty_level = st.selectbox("Enter the difficulty level:", ["Easy", "Medium", "Hard"])
    question_type = st.selectbox("Enter the question type:", ["MCQs", "True/False", "Fill-in-the-Blank", "Text-Based"])

    if st.button("Generate Questions"):
        context = result

        if question_type == "MCQs":
            questions = mcqs(num_questions, difficulty_level, context)
        elif question_type == "Fill-in-the-Blank":
            questions = fill_in_the_blanks(num_questions, difficulty_level, context)
        elif question_type == "True/False":
            questions = true_or_false(num_questions, difficulty_level, context)
        elif question_type == "Text-Based":
            questions = text_based(num_questions, difficulty_level, context)

        st.subheader(f"Generated {question_type} Questions")
        st.text(questions)
    