from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from starlette.middleware.wsgi import WSGIMiddleware
from api import app as flask_app  # Assurez-vous que le chemin est correct

# Création de l'application FastAPI
app_fastapi = FastAPI(title="Documentation de l'API")

# Définir un modèle Pydantic pour la documentation des réponses
class HelloResponse(BaseModel):
    message: str

# Définir les routes dans FastAPI pour la documentation
router = APIRouter()

@router.get("/hello", response_model=HelloResponse)
def read_hello():
    return {"message": "Hello, world!"}

app_fastapi.include_router(router)

# Intégrer l'application Flask en utilisant WSGIMiddleware
app_fastapi.mount("/", WSGIMiddleware(flask_app))
