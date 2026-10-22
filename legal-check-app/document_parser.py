import fitz  # PyMuPDF
from docx import Document

def extract_text(uploaded_file, file_type: str) -> str:
    if file_type == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file_type == "docx":
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")