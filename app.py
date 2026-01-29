import streamlit as st
import pandas as pd
from sqlalchemy.orm import Session
import database

# Configuración de la página
st.set_page_config(page_title="Mi Portafolio", layout="wide")

st.title("📊 Seguimiento de Acciones y Divisas")

# Conexión a la base de datos
db = database.SessionLocal()

