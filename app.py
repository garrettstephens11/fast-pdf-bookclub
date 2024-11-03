import os
import uuid
import openai
import fitz  # PyMuPDF
import re
import nltk
from fastapi import FastAPI, Request, Form, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import logging
from dotenv import load_dotenv
import json
import asyncio
from typing import Dict, List
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

app = FastAPI()

# Configure OpenAI API
openai.organization = os.getenv('OPENAI_ORG_ID')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database setup
DATABASE_URL = "sqlite:///./pdf_bookclub.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

class Segment(Base):
    __tablename__ = "segments"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    index = Column(Integer)
    text = Column(Text)

class Discussion(Base):
    __tablename__ = "discussions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    segment_index = Column(Integer)
    discussion_num = Column(Integer)
    text = Column(Text)
    audio_generated = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def clean_text(text):
    # Clean the text by removing extra whitespaces and correcting punctuation spacing
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = re.sub(r' \.', '.', cleaned_text)
    cleaned_text = re.sub(r' ,', ',', cleaned_text)
    cleaned_text = re.sub(r' !', '!', cleaned_text)
    cleaned_text = re.sub(r' \?', '?', cleaned_text)
    cleaned_text = re.sub(r' ;', ';', cleaned_text)
    cleaned_text = re.sub(r' :', ':', cleaned_text)
    return cleaned_text

def extract_pdf_text(pdf_data):
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    sentences = nltk.sent_tokenize(text)
    paragraphs = [' '.join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
    text = '\n\n'.join(paragraphs)
    text = text.replace(' .', '.')
    return clean_text(text)

def parse_discussion(discussion_text):
    pattern = r'(Person [ABC]|Teacher):\s*(.*?)(?=(?:\nPerson [ABC]|Teacher:|\Z))'
    matches = re.findall(pattern, discussion_text, re.DOTALL)
    discussion_lines = []
    for speaker, text in matches:
        discussion_lines.append({'speaker': speaker, 'text': text.strip()})
    return discussion_lines

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    pdf_data = await pdf_file.read()
    text = extract_pdf_text(pdf_data)
    tokens = nltk.word_tokenize(text)
    segments = [' '.join(tokens[i:i+2000]) for i in range(0, len(tokens), 2000)]
    segments = [clean_text(segment) for segment in segments]
    session_id = str(uuid.uuid4())

    # Store segments in the database
    db = SessionLocal()
    for idx, segment_text in enumerate(segments):
        segment = Segment(session_id=session_id, index=idx, text=segment_text)
        db.add(segment)
    db.commit()
    db.close()

    # Redirect to the first segment
    return RedirectResponse(url=f"/segment/{session_id}/0", status_code=303)

@app.get("/segment/{session_id}/{segment_index}", response_class=HTMLResponse)
async def display_segment(request: Request, session_id: str, segment_index: int):
    db = SessionLocal()
    segment = db.query(Segment).filter_by(session_id=session_id, index=segment_index).first()
    total_segments = db.query(Segment).filter_by(session_id=session_id).count()
    db.close()
    if not segment:
        return HTMLResponse(content="Segment not found", status_code=404)
    return templates.TemplateResponse("segment.html", {
        "request": request,
        "session_id": session_id,
        "segment_index": segment_index,
        "segment": segment.text,
        "total_segments": total_segments
    })

@app.post("/generate")
async def generate(request: Request, session_id: str = Form(...), segment_index: int = Form(...), relation_text: str = Form(None), additional_turn: str = Form('false')):
    db = SessionLocal()
    segment = db.query(Segment).filter_by(session_id=session_id, index=segment_index).first()
    if not segment:
        db.close()
        return JSONResponse(content={'error': 'Segment not found'}, status_code=404)
    
    # Retrieve previous discussions
    discussions = db.query(Discussion).filter_by(session_id=session_id, segment_index=segment_index).order_by(Discussion.discussion_num).all()

    if additional_turn == 'true' and discussions:
        previous_discussion_texts = [disc.text for disc in discussions]
        previous_discussion = '\n\n'.join(previous_discussion_texts)

        prompt = f"""
        Continue the following book club discussion between three people about the following section of a book: {segment.text}.

        Previous discussion:
        {previous_discussion}

        Now, a Teacher joins the discussion. The Teacher's role is to challenge the participants to be more specific in drawing from the text, encouraging them to make cross comparisons between fragments within the text and the full passage. If the participants are loosely summarizing based on simple opinions, the Teacher should guide them to delve deeper.

        Please continue the discussion, including the Teacher and the three participants. The discussion should be formatted as follows:

        Person A: [Person A's comment]
        Person B: [Person B's comment]
        Person C: [Person C's comment]
        Teacher: [Teacher's comment]
        """

        if relation_text:
            prompt += f" Next, have the group participants relate the reading to {relation_text}."
    else:
        prompt = f"""
        Can you generate for me a short book club discussion between three people about the following section of a book: {segment.text}.

        The discussion should be formatted as follows:
        Person A: [Person A's comment]
        Person B: [Person B's comment]
        Person C: [Person C's comment]

        This discussion should involve at least two "turns" per discussion participant in this discussion. The discussion should address the book club question: 'What is a random word or phrase that stood out to you in reading this text? What does that word or phrase bring to mind for you? Then, relate your thought back to the passage's message.'
        """

        if relation_text:
            prompt += f" Next, have the group participants relate the reading to {relation_text}."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    discussion_text = response.choices[0].message['content']
    logging.info(f"Raw discussion output: {discussion_text}")

    # Parse the discussion
    discussion_lines = parse_discussion(discussion_text)

    # Store the discussion in the database
    discussion_num = len(discussions)
    new_discussion = Discussion(
        session_id=session_id,
        segment_index=segment_index,
        discussion_num=discussion_num,
        text=discussion_text,
        audio_generated=False
    )
    db.add(new_discussion)
    db.commit()
    db.close()

    return JSONResponse(content={"discussion": discussion_text, "discussion_lines": discussion_lines})

@app.post("/generate_audio")
async def generate_audio(request: Request, session_id: str = Form(...), segment_index: int = Form(...), discussion_num: int = Form(...)):
    # Placeholder for audio generation logic
    # You can implement your TTS logic here
    # For now, we'll just return a success message
    return JSONResponse(content={"message": "Audio generated successfully"})

@app.post("/format_rough")
async def format_text_rough(segment: str = Form(...)):
    sentences = segment.split('.')
    rough_formatted_text = ''

    for i in range(len(sentences)):
        rough_formatted_text += sentences[i].strip() + '.'
        if (i + 1) % 3 == 0 and i < len(sentences) - 1:  # every third sentence
            rough_formatted_text += '<br><br>'

    logging.info(f"Rough formatted text output: {rough_formatted_text}")
    return JSONResponse(content={'formatted_text': rough_formatted_text})

@app.get("/chat/{session_id}/{segment_index}", response_class=HTMLResponse)
async def chat(request: Request, session_id: str, segment_index: int):
    db = SessionLocal()
    segment = db.query(Segment).filter_by(session_id=session_id, index=segment_index).first()
    db.close()
    if not segment:
        return HTMLResponse(content="Segment not found", status_code=404)
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": session_id,
        "segment_index": segment_index,
        "segment": segment.text
    })

@app.websocket("/ws/chat/{session_id}/{segment_index}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, segment_index: int):
    await websocket.accept()
    db = SessionLocal()
    segment = db.query(Segment).filter_by(session_id=session_id, index=segment_index).first()
    db.close()
    if not segment:
        await websocket.send_json({"error": "Segment not found"})
        await websocket.close(code=1008)
        return

    conversation = [
        {"role": "system", "content": f"You are a helpful assistant knowledgeable about the following passage:\n{segment.text}"}
    ]

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            user_message = message.get('message')
            if user_message:
                conversation.append({"role": "user", "content": user_message})
                # Call OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=conversation,
                )
                bot_message = response.choices[0].message['content']
                conversation.append({"role": "assistant", "content": bot_message})
                await websocket.send_json({"message": bot_message})
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await websocket.close(code=1008)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)
