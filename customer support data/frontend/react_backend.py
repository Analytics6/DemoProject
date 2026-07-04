from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from frontend.app_backend import authenticate_user, get_dashboard_stats, initialize_database, list_tickets

initialize_database()

app = FastAPI(title="Retail Support API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/health')
def health() -> dict:
    return {"status": "ok", "stats": get_dashboard_stats()}


@app.post('/auth')
def auth(payload: dict):
    username = payload.get('username', '')
    password = payload.get('password', '')
    if not authenticate_user(username, password):
        return JSONResponse(status_code=401, content={"error": "Invalid credentials"})
    from frontend.app_backend import get_user_profile
    profile = get_user_profile(username)
    return {"user": profile, "stats": get_dashboard_stats(), "tickets": list_tickets()}
