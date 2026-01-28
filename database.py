from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Definimos la ubicación del archivo de base de datos
SQLALCHEMY_DATABASE_URL = "sqlite:///./acciones.db"

# Creamos el motor de conexión
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# Creamos una clase para manejar sesiones (conexiones temporales)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clase base para nuestras tablas
Base = declarative_base()

# Definición de la tabla
class Transaccion(Base):
    __tablename__ = "transacciones"

    id = Column(Integer, primary_key=True, index=True)
    fecha = Column(String)              # ISO 8601: YYYY-MM-DD
    broker = Column(String)             # Ej: "Rational", "BancoEstado"
    tipo_activo = Column(String)        # "accion" o "divisa"
    tipo_transaccion = Column(String)   # "compra" o "venta"
    ticker = Column(String)             # Ej: "AAPL" o "USD/CLP"
    monto_total = Column(Float)         # Dinero total en DÓLARES (USD)
    precio_unitario = Column(Float)     # Precio de 1 accion (USD) o valor del dolar (CLP)
    cantidad = Column(Float)            # Cantidad calculada de activos

class IngresoPasivo(Base):
    __tablename__ = "ingresos_pasivos"

    id = Column(Integer, primary_key=True, index=True)
    fecha = Column(String)          # ISO 8601: YYYY-MM-DD
    broker = Column(String)         # Ej: "Rational"
    tipo_ingreso = Column(String)   # "dividendo" o "interes"
    ticker = Column(String)         # Ej: "AAPL" o "Cuenta Premium"
    monto = Column(Float)           # Monto antes de impuestos (USD)