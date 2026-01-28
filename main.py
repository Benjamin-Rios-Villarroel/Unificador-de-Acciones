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

@app.post("/registrar")
def registrar_transaccion(
    fecha: str,
    broker: str,
    tipo_activo: str,
    tipo_transaccion: str,
    ticker: str,
    monto_total: float,
    precio_unitario: float,
    cantidad: float,
    db: Session = Depends(get_db)
):
    nueva_transaccion = database.Transaccion(
        fecha=fecha,
        broker=broker,
        tipo_activo=tipo_activo,
        tipo_transaccion=tipo_transaccion,
        ticker=ticker,
        monto_total=monto_total,
        precio_unitario=precio_unitario,
        cantidad=cantidad
    )
    db.add(nueva_transaccion)
    db.commit()
    db.refresh(nueva_transaccion)
    return {"mensaje": "Registro exitoso", "id": nueva_transaccion.id}

# Ver todas las transacciones
@app.get("/transacciones")
def obtener_todas(db: Session = Depends(get_db)):
    return db.query(database.Transaccion).all()