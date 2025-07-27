import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def process_pdfs(data_dir="data/pdfs"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Please add your PDFs to {data_dir}")
        return None

    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            text = extract_text_from_pdf(filepath)
            documents.append(text)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.create_documents(documents)
    return chunks