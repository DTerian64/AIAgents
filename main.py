# main.py
#pip install fastapi uvicorn jinja2
#pip install azure-ai-inference
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from Agent.ministral_agent import get_agent_response

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")
    answer = get_agent_response(question)
    return JSONResponse({"answer": answer})
