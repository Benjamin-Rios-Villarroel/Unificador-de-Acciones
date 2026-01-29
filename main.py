from fastapi import FastAPI, Depends
import yfinance as yf
from sqlalchemy.orm import Session
import database

database.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

# Función para obtener la conexión a la base de datos
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def inicio():
    return {"mensaje": "Bienvenido a mi unificador de acciones"}
