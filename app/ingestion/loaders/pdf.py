import io
import logfire
from pypdf import PdfReader, PdfWriter
from google.cloud import documentai
from app.config import settings

client = documentai.DocumentProcessorServiceClient()
MAX_PAGE_PER_REQUEST = 15


def parse_pdf(file_path: str):
    """
    Parse PDF files using Google Document AI
    Split PDF into chunks of 15 pages to avoid "Document too large" error
    """
    with logfire.span("PDF Processing with Document AI (Google)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"Total pages: {total_pages}")

            name = client.processor_path(
                settings.PROJECT_ID, 
                settings.GCP_DOC_AI_LOCATION, 
                settings.GCP_DOC_AI_PROCESSOR_ID)

            full_text = ""

            # if the pages are less than or equal to MAX_PAGE_PER_REQUEST, process as single batch
            if total_pages <= MAX_PAGE_PER_REQUEST:
                logfire.info("Processing PDF as single batch")
                with open(file_path, "rb") as file:
                    image_content = file.read()
                full_text = _process_document_chunk(name, image_content)
            else:
                logfire.info(f"Processing PDF in chunks of {MAX_PAGE_PER_REQUEST} pages")
                texts = []
                for start_page in range(0, total_pages, MAX_PAGE_PER_REQUEST):
                    end_page = min(start_page + MAX_PAGE_PER_REQUEST, total_pages)
                    logfire.info(f"Processing pages {start_page + 1} to {end_page}")
                    
                    # Create a new PDF containing only the pages in this chunk
                    writer = PdfWriter()
                    for page_num in range(start_page, end_page):
                        writer.add_page(reader.pages[page_num])
                    
                    # Write the PDF chunk to a bytes buffer
                    pdf_buffer = io.BytesIO()
                    writer.write(pdf_buffer)
                    pdf_bytes = pdf_buffer.getvalue()
                    
                    # Process the chunk
                    chunk_text = _process_document_chunk(name, pdf_bytes)
                    texts.append(chunk_text)
                
                full_text = "\n".join(texts)

            if not full_text.strip():
                logfire.warning(f"Document AI returned empty text for {file_path}")
                return None
            else: 
                logfire.info(f"Document AI successfully parsed {len(full_text)} characters")

            return full_text
        except Exception as e:
            logfire.error(f"Error parsing PDF: {e}")
            raise e
            

def _process_document_chunk(name: str, image_content: bytes) -> str:
    """Helper function to send a specific byte chunk to Document AI"""
    raw_document = documentai.RawDocument(
        content=image_content, 
        mime_type="application/pdf"
    )

    request = documentai.ProcessRequest(
        name=name, 
        raw_document=raw_document
    )

    result = client.process_document(request=request)
    return result.document.text