# main.py
#pip install fastapi uvicorn jinja2
#pip install azure-ai-inference
#pip install python-jose
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Depends, HTTPException, status, Request
from jose import jwt
import requests

from Agent.ministral_agent import get_agent_response

import time



TENANT_ID = "4d5f34d3-d97b-40c7-8704-edff856d3654"
CLIENT_ID = "177da031-26fa-448a-8521-1d9bedde86d3"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
JWKS_URL = f"{AUTHORITY}/discovery/v2.0/keys"


_jwks_cache = None
_jwks_last_fetch = 0

def get_jwks():
    global _jwks_cache, _jwks_last_fetch
    # Refresh every 24h
    if not _jwks_cache or time.time() - _jwks_last_fetch > 86400:
        r = requests.get(JWKS_URL)
        r.raise_for_status()
        _jwks_cache = r.json()
        _jwks_last_fetch = time.time()
    return _jwks_cache


def verify_token(request: Request):
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    try:
        # Validate signature
        header = jwt.get_unverified_header(token)
        jwks = get_jwks()  # <-- lazy loaded + cached!
        key = next(k for k in jwks["keys"] if k["kid"] == header["kid"])
        payload = jwt.decode(
            token,
            key,
            algorithms=header["alg"],
            audience=CLIENT_ID,
            issuer=f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"
        )
        return payload
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(request: Request, token_data=Depends(verify_token)):
    data = await request.json()
    question = data.get("question", "")
    answer = get_agent_response(question)
    return JSONResponse({"answer": answer})

# Health check endpoint for Azure
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Catch-all route for SPA - this should be LAST
@app.get("/{path:path}")
async def catch_all(request: Request, path: str):
    # Don't intercept API routes
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    # Return index.html for all other routes (SPA routing)
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
