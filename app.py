import streamlit as st
import pandas as pd
from sqlalchemy.orm import Session
import database

# Configuración de la página
st.set_page_config(page_title="Mi Portafolio", layout="wide")

st.title("📊 Seguimiento de Acciones y Divisas")

# Conexión a la base de datos
db = database.SessionLocal()

# Función para cargar datos
def cargar_datos():
    query = db.query(database.Transaccion).all()
    # Convertimos los objetos de la base de datos a un DataFrame de Pandas (tabla)
    df = pd.DataFrame([u.__dict__ for u in query])
    if not df.empty:
        df = df.drop(columns=['_sa_instance_state']) # Limpieza técnica de SQLAlchemy
    return df

datos = cargar_datos()

# Interfaz de usuario
if not datos.empty:
    st.subheader("Tus Transacciones")
    # Mostramos la tabla interactiva
    st.dataframe(datos, use_container_width=True)
else:
    st.info("Aún no hay transacciones registradas.")

# Formulario para agregar nueva transacción
with st.sidebar:
    st.header("Registrar Operación")
    with st.form("nueva_transaccion"):
        fecha = st.date_input("Fecha")
        broker = st.selectbox("Broker", ["Racional", "Zesty", "BancoEstado", "Otro"])
        tipo_activo = st.radio("Tipo de Activo", ["accion", "divisa"])
        tipo_transaccion = st.radio("Operación", ["compra", "venta"])
        ticker = st.text_input("Ticker (ej: AAPL o USD/CLP)")
        monto = st.number_input("Monto Total (USD)", min_value=0.0)
        precio = st.number_input("Precio Unitario", min_value=0.0)
        cantidad = st.number_input("Cantidad", min_value=0.0)
        
        submit = st.form_submit_button("Guardar")
        
        if submit:
            nueva = database.Transaccion(
                fecha=str(fecha),
                broker=broker,
                tipo_activo=tipo_activo,
                tipo_transaccion=tipo_transaccion,
                ticker=ticker.upper(),
                monto_total=monto,
                precio_unitario=precio,
                cantidad=cantidad
            )
            db.add(nueva)
            db.commit()
            st.success("¡Registrado!")
            st.rerun() # Recarga la página para ver los cambios