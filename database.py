from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Definimos la ubicación del archivo de base de datos
SQLALCHEMY_DATABASE_URL = "sqlite:///./inversiones.db"

# Creamos el motor de conexión
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# Creamos una clase para manejar sesiones (conexiones temporales)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clase base para nuestras tablas
Base = declarative_base()

# Tabla para las transacciones de acciones
class Transacciones_acciones(Base):
    __tablename__ = "transacciones_acciones"
    id = Column(Integer, primary_key=True, index=True)
    fecha = Column(String)              # ISO 8601: YYYY-MM-DD
    broker = Column(String)             # Ej: "Racional", "Zesty"
    tipo_transaccion = Column(String)   # "compra" o "venta"
    ticker = Column(String)             # Ej: "AAPL"
    monto_total = Column(Float)         # Dinero total en DÓLARES (USD)
    precio = Column(Float)              # Precio de 1 accion (USD)
    cantidad = Column(Float)            # Cantidad de acciones

# Tabla para las transacciones de dólares y pesos
class Trasacciones_divisas(Base):
    __tablename__ = "transacciones_divisas"
    id = Column(Integer, primary_key=True, index=True)
    fecha = Column(String)              # ISO 8601: YYYY-MM-DD
    broker = Column(String)             # Ej: "Racional", "Zesty"
    tipo_transaccion = Column(String)   # "compra" o "venta"
    monto_total = Column(Float)         # Dinero total en DÓLARES (CLP)
    precio = Column(Float)              # Precio de 1 dólar (CLP)
    cantidad = Column(Float)            # Cantidad de dólares (USD)

# Tabla para los dividendos e interéses
class Ingreso_pasivo(Base):
    __tablename__ = "ingresos_pasivos"
    id = Column(Integer, primary_key=True, index=True)
    fecha = Column(String)          # ISO 8601: YYYY-MM-DD
    broker = Column(String)         # Ej: "Racional", "Zesty"
    tipo_ingreso = Column(String)   # "dividendo" o "interés"
    ticker = Column(String)         # Ej: "AAPL" o "boost" o "interés"
    monto = Column(Float)           # Monto (USD)

# Tabla para los montos a final de mes
class Resumen_mensual(Base):
    __tablename__ = "resumen_mensual"
    id = Column(Integer, primary_key=True, index=True)
    fecha = Column(String)          # YYYY-MM
    broker = Column(String)         # Ej: "Racional", "Zesty"
    monto_pesos = Column(Float)     # Cantidad mensual en pesos
    monto_dolares = Column(Float)   # Cantidad mensual en dólares

class Historico_diario(Base):
    __tablename__ = "historico_diario"
    id = Column(Integer, primary_key=True, index=True)
    fecha = Column(String)          # YYYY-MM-DD
    broker = Column(String)         # Ej: "Racional", "Zesty"
    ticker = Column(String)         # Ej: "AAPL"
    precio_cierre = Column(Float)   # Cierre diario 
    cantidad = Column(Float)        # Cantidad de acciones en portafolio
    valor = Column(Float)           # Valor de las acciones