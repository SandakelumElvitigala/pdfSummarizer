import os
import tempfile
import logging
import json
import re
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pdfplumber
from typing import Optional, List
import requests
from dotenv import load_dotenv
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fastapi.responses import FileResponse
import concurrent.futures

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak
from reportlab.platypus import Frame, PageTemplate
from markdown2 import markdown
from bs4 import BeautifulSoup

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# === Pydantic models ===
class SummaryRequest(BaseModel):
    max_length: int = Field(default=500)
    min_length: int = Field(default=100)
    focus_areas: Optional[str] = None

class SummaryResponse(BaseModel):
    summary: str
    page_count: int
    word_count: int
    extraction_status: str

# === Helper: Extract text from PDF ===
def extract_text_from_pdf(file_path):
    logger.info(f"Extracting text from {file_path}")
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        word_count = len(text.split())
        return text, page_count, word_count, "Success"
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return "", 0, 0, f"Error extracting text: {str(e)}"

# === Helper: Simple entity highlighting ===
def simple_highlight_entities(text):
    def highlight_orgs(m): return f"**{m.group(0)}**"
    def highlight_dates(m): return f"**{m.group(0)}**"
    text = re.sub(r'\b[A-Z][a-z]+(\s[A-Z][a-z]+)+\b', highlight_orgs, text)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b',
    ]
    for pattern in date_patterns:
        text = re.sub(pattern, highlight_dates, text)
    return text

# === Chunk text helper ===
def chunk_text(text: str, max_words=800) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# === Process a single chunk ===
def process_chunk(chunk_data):
    chunk, idx, total_chunks, max_length, focus_areas = chunk_data
    logger.info(f"Summarizing chunk {idx+1}/{total_chunks}...")
    attempt = 0
    max_retries = 5
    delay_between_retries = 1.5
    
    while attempt < max_retries:
        try:
            focus_instruction = f"Focus on: {focus_areas}." if focus_areas else ""
            prompt = f"""
Summarize the following text concisely. Use **markdown** to bold:
- Organization names
- Dates
- Locations
{focus_instruction}

TEXT TO SUMMARIZE:
{chunk}
            """
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are a professional summarizer. Use markdown for key entities."},
                    {"role": "user", "content": prompt.strip()}
                ],
                "temperature": 0.3,
                "max_tokens": max_length * 2
            }

            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                summary = response.json()["choices"][0]["message"]["content"]
                if "**" not in summary:
                    summary = simple_highlight_entities(summary)
                return summary
            elif response.status_code == 429:
                wait_time = delay_between_retries * (attempt + 1)
                logger.warning(f"Rate limited on chunk {idx+1}, retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
                attempt += 1
            else:
                logger.error(f"Groq API error (chunk {idx+1}): {response.status_code} - {response.text}")
                return f"❌ Chunk {idx+1} summary failed."
                
        except Exception as e:
            logger.error(f"Chunk summarization error: {str(e)}")
            return f"❌ Chunk {idx+1} failed: {e}"
            
    return f"❌ Chunk {idx+1} skipped after {max_retries} retries."

# === Updated summarizer with parallel processing ===
def meaningful_summarize(text, max_length=500, focus_areas=None):
    logger.info("Generating meaningful summary via Groq API with parallel processing...")

    if not text.strip():
        return "**No text to summarize.**"

    # Create all chunks first
    chunks = chunk_text(text)
    logger.info(f"Document split into {len(chunks)} chunks")
    
    # Prepare data for parallel processing
    chunk_data = [(chunk, idx, len(chunks), max_length, focus_areas) 
                  for idx, chunk in enumerate(chunks)]
    
    # Process chunks in parallel
    all_summaries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_chunk = {executor.submit(process_chunk, data): data for data in chunk_data}
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_data = future_to_chunk[future]
            try:
                summary = future.result()
                all_summaries.append((chunk_data[1], summary))  # Store (index, summary)
            except Exception as e:
                idx = chunk_data[1]
                logger.error(f"Executor exception for chunk {idx+1}: {str(e)}")
                all_summaries.append((idx, f"❌ Chunk {idx+1} failed with executor error"))
    
    # Sort summaries by their original index and extract just the summaries
    all_summaries.sort(key=lambda x: x[0])
    combined_summary = "\n\n".join([summary for _, summary in all_summaries])

    logger.info("Refining combined summary to final word count...")
    try:
        refined_summary = refine_summary_to_length(combined_summary, max_length=max_length)
        final_clipped = clip_to_max_words(refined_summary, max_words=max_length)
        return final_clipped.replace(". ", ".\n\n")
    except Exception as e:
        logger.error(f"Final summarization exception: {str(e)}")
        return clip_to_max_words(combined_summary, max_words=max_length)

# === Refine summary to enforce length ===
def refine_summary_to_length(summary_text, max_length=500):
    logger.info("Refining summary to enforce final word limit...")

    summary_chunks = chunk_text(summary_text, max_words=800)
    refined_chunks = []

    for idx, chunk in enumerate(summary_chunks):
        logger.info(f"Refining chunk {idx+1}/{len(summary_chunks)}...")
        retries = 3
        attempt = 0

        while attempt < retries:
            try:
                prompt = f"""
Refine the following text into a concise, clear summary. Target: **exactly {max_length} words**.
Use markdown for important entities (organizations, dates, locations).
Do not exceed the word count.

TEXT TO REFINE:
{chunk}
                """
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a professional summarizer. Format using markdown for entities."},
                        {"role": "user", "content": prompt.strip()}
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_length * 2
                }

                response = requests.post(GROQ_API_URL, headers=headers, json=payload)

                if response.status_code == 200:
                    refined = response.json()["choices"][0]["message"]["content"]
                    refined_chunks.append(refined)
                    break

                elif response.status_code == 429:
                    try:
                        response_data = response.json()
                        msg = response_data.get("error", {}).get("message", "")
                        wait_match = re.search(r"try again in ([\d.]+)s", msg)
                        wait_time = float(wait_match.group(1)) if wait_match else 10.0
                        logger.warning(f"Rate limit hit on refinement chunk {idx+1}, sleeping for {wait_time:.2f}s...")
                        time.sleep(wait_time)
                        attempt += 1
                    except Exception as e:
                        logger.warning(f"Could not parse retry time: {e}")
                        time.sleep(10)
                        attempt += 1
                else:
                    logger.warning(f"Refining chunk {idx+1} failed: {response.status_code} - {response.text}")
                    refined_chunks.append(chunk)
                    break

            except Exception as e:
                logger.warning(f"Refining chunk {idx+1} error: {e}")
                refined_chunks.append(chunk)
                break

        else:
            logger.warning(f"Refining chunk {idx+1} failed after {retries} retries.")
            refined_chunks.append(chunk)

    combined_refined = "\n\n".join(refined_chunks)

    if len(combined_refined.split()) > max_length * 1.2:
        logger.info("Final compression pass to enforce strict word limit...")
        try:
            final_prompt = f"""
Compress the following summary into **exactly {max_length} words**. Use markdown for important entities.
Avoid exceeding the word count.

TEXT TO COMPRESS:
{combined_refined}
            """
            payload["messages"][1]["content"] = final_prompt.strip()
            final_response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            if final_response.status_code == 200:
                final_summary = final_response.json()["choices"][0]["message"]["content"]
                return clip_to_max_words(final_summary, max_words=max_length).strip()
        except Exception as e:
            logger.warning(f"Final compression failed: {e}")

    return clip_to_max_words(combined_refined, max_words=max_length).strip()

# === Final clip ===
def clip_to_max_words(text, max_words=500):
    words = text.split()
    if len(words) <= max_words:
        return text

    logger.info(f"Final summary too long ({len(words)} words). Clipping to {max_words} words.")
    clipped = ' '.join(words[:max_words])

    # Cleanly close the last sentence (if mid-sentence)
    if not clipped.endswith(('.', '!', '?')):
        clipped = re.sub(r'[^\.!\?]*$', '', clipped).strip()
        if not clipped.endswith(('.', '!', '?')):
            clipped += "..."

    return clipped

# === Generate PDF summary ===
def generate_summary_pdf(summary_text: str, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    normal_style.fontSize = 11
    normal_style.leading = 14
    normal_style.spaceAfter = 10
    normal_style.alignment = TA_LEFT

    # Convert markdown to HTML and then parse
    html = markdown(summary_text)
    soup = BeautifulSoup(html, "html.parser")

    flowables = []

    for element in soup.descendants:
        if element.name == 'p':
            para = Paragraph(str(element), normal_style)
            flowables.append(para)
            flowables.append(Spacer(1, 10))

    doc.build(flowables)

# === FastAPI endpoint ===
@app.post("/summarize", response_model=SummaryResponse)
async def summarize_pdf(file: UploadFile = File(...), max_length: int = Form(500), focus_areas: str = Form(None)):
    logger.info(f"Received file: {file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        text, page_count, word_count, status = extract_text_from_pdf(temp_path)
        if "Error" in status:
            raise HTTPException(status_code=500, detail=status)

        summary = meaningful_summarize(text, max_length=max_length, focus_areas=focus_areas)
        summary = clip_to_max_words(summary, max_words=max_length).strip()

        return SummaryResponse(
            summary=summary,
            page_count=page_count,
            word_count=len(summary.split()),
            extraction_status=status
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

# === FastAPI endpoint for PDF generation ===
@app.post("/summarize-to-pdf")
async def summarize_and_generate_pdf(
    file: UploadFile = File(...),
    max_length: int = Form(500),
    focus_areas: str = Form(None)
):
    logger.info(f"Received file: {file.filename}")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")

    temp_pdf_path = None
    summary_pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_pdf_path = temp_file.name
            temp_file.write(await file.read())

        text, page_count, word_count, status = extract_text_from_pdf(temp_pdf_path)
        if "Error" in status:
            raise HTTPException(status_code=500, detail=status)

        summary = meaningful_summarize(text, max_length=max_length, focus_areas=focus_areas)
        summary = clip_to_max_words(summary, max_words=max_length).strip()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as summary_pdf:
            summary_pdf_path = summary_pdf.name
            generate_summary_pdf(summary, summary_pdf_path)

        return FileResponse(summary_pdf_path, filename="summarized_output.pdf", media_type="application/pdf")

    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

# === Health Check ===
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "PDF Summarization API is running",
        "groq_api_configured": bool(GROQ_API_KEY)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)