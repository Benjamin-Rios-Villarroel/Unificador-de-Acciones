import streamlit as st
import pandas as pd
import database
import yfinance as yf
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from sqlalchemy import func

# Configuración de página y base de datos
st.set_page_config(page_title="Unificador de Acciones", layout="wide")
database.Base.metadata.create_all(bind=database.engine)
db = database.SessionLocal()

# --- FUNCIÓN HELPER PARA CARGAR TABLAS ---
def cargar_tabla(modelo):
    query = db.query(modelo).all()
    if not query:
        return pd.DataFrame()
    # Convertimos a DataFrame y eliminamos metadatos internos de SQLAlchemy
    df = pd.DataFrame([u.__dict__ for u in query])
    if '_sa_instance_state' in df.columns:
        df = df.drop(columns=['_sa_instance_state'])
    return df

st.title("📂 Gestión de Mi Portafolio")

# --- PESTAÑAS PRINCIPALES ---
tab_acciones, tab_divisas, tab_pasivos, tab_resumen, tab_historico = st.tabs([
    "📈 Acciones", "💵 Divisas (USD/CLP)", "💰 Ingresos Pasivos", "🗓️ Resumen Mensual", "📜 Histórico Diario"
])

# --- 1. SECCIÓN DE ACCIONES ---
with tab_acciones:
    st.header("Transacciones de Acciones")
    with st.expander("➕ Registrar Nueva Compra/Venta"):
        with st.form("form_acciones"):
            col1, col2, col3 = st.columns(3)
            with col1:
                fecha = st.date_input("Fecha", key="acc_fecha")
                ticker = st.text_input("Ticker (ej: AAPL)").upper()
            with col2:
                tipo = st.selectbox("Operación", ["compra", "venta"])
                broker = st.selectbox("Broker", ["Racional", "Zesty", "Fintual", "Otro"])
            with col3:
                monto_total = st.number_input("Monto Total (USD)", min_value=0.0)
                precio = st.number_input("Precio por Acción (USD)", min_value=0.0)
                cantidad = st.number_input("Cantidad de Acciones", min_value=0.0)
            
            if st.form_submit_button("Guardar Transacción"):
                nueva_trans = database.Transacciones_acciones(
                    fecha=str(fecha), broker=broker, tipo_transaccion=tipo,
                    ticker=ticker, monto_total=monto_total, precio=precio, cantidad=cantidad
                )
                db.add(nueva_trans)
                db.commit()
                st.success(f"Registrada {tipo} de {ticker}")
                st.rerun()
    
    st.subheader("Registros Existentes")
    st.dataframe(cargar_tabla(database.Transacciones_acciones), use_container_width=True)

# --- 2. SECCIÓN DE DIVISAS ---
with tab_divisas:
    st.header("Compra y Venta de Dólares")
    with st.expander("➕ Registrar Cambio de Divisa"):
        with st.form("form_divisas"):
            c1, c2, c3 = st.columns(3)
            with c1:
                f_div = st.date_input("Fecha")
                tipo_d = st.selectbox("Tipo de Transacción", ["compra", "venta"])
                broker_d = st.selectbox("Broker", ["Racional", "Zesty", "BancoEstado", "Otro"])
            with c2:
                monto_clp = st.number_input("Valor Total en CLP", min_value=0.0)
                tipo_cambio = st.number_input("Tipo de Cambio (CLP/USD)", min_value=0.0)
            with c3:
                total_usd = st.number_input("Total USD Obtenidos/Vendidos", min_value=0.0)
            
            if st.form_submit_button("Registrar Divisa"):
                nueva_div = database.Trasacciones_divisas(
                    fecha=str(f_div), broker=broker_d, tipo_transaccion=tipo_d,
                    monto_total=monto_clp, precio=tipo_cambio, cantidad=total_usd
                )
                db.add(nueva_div)
                db.commit()
                st.info("Divisa registrada")
                st.rerun()
    
    st.subheader("Registros de Divisas")
    st.dataframe(cargar_tabla(database.Trasacciones_divisas), use_container_width=True)

# --- 3. INGRESOS PASIVOS ---
with tab_pasivos:
    st.header("Dividendos e Intereses")
    with st.expander("➕ Registrar Ingreso"):
        with st.form("form_pasivos"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                fecha_p = st.date_input("Fecha de Recepción")
                broker_p = st.selectbox("Broker", ["Racional", "Zesty", "Fintual", "Otro"])
            with col_b:
                tipo_p = st.selectbox("Tipo de Ingreso", ["dividendo", "interés"])
                ticker_p = st.text_input("Empresa/Fuente")
            with col_c:
                monto_p = st.number_input("Monto Neto (USD)", min_value=0.0)
            
            if st.form_submit_button("Añadir Ingreso"):
                nuevo_p = database.Ingreso_pasivo(
                    fecha=str(fecha_p), broker=broker_p, tipo_ingreso=tipo_p,
                    ticker=ticker_p.upper(), monto=monto_p
                )
                db.add(nuevo_p)
                db.commit()
                st.success("Ingreso registrado")
                st.rerun()
    
    st.subheader("Historial de Ingresos")
    st.dataframe(cargar_tabla(database.Ingreso_pasivo), use_container_width=True)

# --- 4. RESUMEN MENSUAL ---
with tab_resumen:
    st.header("Resúmenes Mensuales")
    df_res = cargar_tabla(database.Resumen_mensual)
    if not df_res.empty:
        st.dataframe(df_res, use_container_width=True)
    
    col_add, col_edit = st.columns(2)
    with col_add:
        st.subheader("➕ Agregar")
        with st.form("add_resumen"):
            f_res = st.text_input("Mes (YYYY-MM)")
            b_res = st.selectbox("Broker", ["Racional", "Zesty", "Fintual"])
            m_clp = st.number_input("Monto Pesos (CLP)", min_value=0.0)
            m_usd = st.number_input("Monto Dólares (USD)", min_value=0.0)
            if st.form_submit_button("Guardar"):
                db.add(database.Resumen_mensual(fecha=f_res, broker=b_res, monto_pesos=m_clp, monto_dolares=m_usd))
                db.commit()
                st.rerun()

# --- 5. HISTÓRICO DIARIO ---
with tab_historico:
    st.header("Histórico Diario de Portafolio")
    st.write("Esta tabla muestra el valor de tus acciones día por día según los registros oficiales.")
    
    df_hist = cargar_tabla(database.Historico_diario)
    if not df_hist.empty:
        # Ordenar por fecha para mejor visualización
        df_hist = df_hist.sort_values(by="fecha", ascending=False)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("Aún no hay datos históricos generados.")