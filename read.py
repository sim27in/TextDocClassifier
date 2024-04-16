from PyPDF2 import PdfReader
from docx import Document


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        if num_pages > 0:
            full_text = ""
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                full_text += text + "\n"
            return full_text
        else:
            return "No pages found in the PDF."

    except Exception as e:
        return f"An error occurred: {str(e)}"


def extract_text_from_doc(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    except Exception as e:
        return f"An error occurred: {str(e)}"