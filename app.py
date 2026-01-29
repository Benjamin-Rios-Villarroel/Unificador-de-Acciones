import streamlit as st
import pandas as pd
import database
from sqlalchemy.orm import Session
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Unificador de Acciones", layout="wide")

# --- CONEXIÓN A DB ---
db = database.SessionLocal()

st.title("📂 Gestión de Mi Portafolio")

# --- PESTAÑAS PRINCIPALES ---
tab_acciones, tab_divisas, tab_pasivos, tab_resumen = st.tabs([
    "📈 Acciones", "💵 Divisas (USD/CLP)", "💰 Ingresos Pasivos", "🗓️ Resumen Mensual"
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

# --- 2. SECCIÓN DE DIVISAS ---
with tab_divisas:
    st.header("Compra y Venta de Dólares")
    with st.expander("➕ Registrar Cambio de Divisa"):
        with st.form("form_divisas"):
            c1, c2, c3 = st.columns(3)
            with c1:
                f_div = st.date_input("Fecha")
                tipo_d = st.selectbox("Tipo de Transacción", ["compra", "venta"])
            with c2:
                monto_clp = st.number_input("Valor Total en CLP", min_value=0.0)
                tipo_cambio = st.number_input("Tipo de Cambio (CLP/USD)", min_value=0.0)
            with c3:
                total_usd = st.number_input("Total USD Obtenidos/Vendidos", min_value=0.0)
            
            if st.form_submit_button("Registrar Divisa"):
                nueva_div = database.Trasacciones_divisas(
                    fecha=str(f_div), tipo_transaccion=tipo_d,
                    monto_total=monto_clp, precio=tipo_cambio, cantidad=total_usd
                )
                db.add(nueva_div)
                db.commit()
                st.info("Divisa registrada en la base de datos")
                st.rerun()

# --- 3. INGRESOS PASIVOS ---
with tab_pasivos:
    st.header("Dividendos e Intereses")
    with st.expander("➕ Registrar Ingreso"):
        with st.form("form_pasivos"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                fecha_p = st.date_input("Fecha de Recepción")
                broker_p = st.selectbox("Broker", ["Racional", "Zesty", "Fintual", "Otro"], key="br_p")
            with col_b:
                tipo_p = st.selectbox("Tipo de Ingreso", ["dividendo", "interés"])
                ticker_p = st.text_input("Empresa/Fuente (ej: AAPL o Billetera)")
            with col_c:
                monto_p = st.number_input("Monto Neto (USD)", min_value=0.0)
            
            if st.form_submit_button("Añadir Ingreso"):
                nuevo_p = database.Ingreso_pasivo(
                    fecha=str(fecha_p), broker=broker_p, tipo_ingreso=tipo_p,
                    ticker=ticker_p.upper(), monto=monto_p
                )
                db.add(nuevo_p)
                db.commit()
                st.success("Ingreso pasivo contabilizado")
                st.rerun()

# --- 4. RESUMEN MENSUAL ---
with tab_resumen:
    st.header("Resúmenes Mensuales")
    
    # Mostrar tabla de resúmenes existentes
    resumenes = db.query(database.Resumen_mensual).all()
    if resumenes:
        df_res = pd.DataFrame([r.__dict__ for r in resumenes]).drop(columns=['_sa_instance_state'])
        st.dataframe(df_res, use_container_width=True)
    
    col_add, col_edit = st.columns(2)
    
    with col_add:
        st.subheader("➕ Agregar Resumen")
        with st.form("add_resumen"):
            f_res = st.text_input("Mes (YYYY-MM)", placeholder="2024-07")
            b_res = st.selectbox("Broker", ["Racional", "Zesty", "Fintual"], key="br_res")
            m_clp = st.number_input("Monto Pesos (CLP)", min_value=0.0)
            m_usd = st.number_input("Monto Dólares (USD)", min_value=0.0)
            if st.form_submit_button("Guardar"):
                nuevo_r = database.Resumen_mensual(fecha=f_res, broker=b_res, monto_pesos=m_clp, monto_dolates=m_usd)
                db.add(nuevo_r)
                db.commit()
                st.rerun()

    with col_edit:
        st.subheader("📝 Modificar Existente")
        if resumenes:
            id_edit = st.selectbox("Seleccionar ID para editar", df_res['id'])
            with st.form("edit_resumen"):
                # Obtener datos actuales
                res_actual = db.query(database.Resumen_mensual).filter_by(id=id_edit).first()
                new_clp = st.number_input("Nuevo Monto Pesos", value=float(res_actual.monto_pesos))
                new_usd = st.number_input("Nuevo Monto Dólares", value=float(res_actual.monto_dolates))
                if st.form_submit_button("Actualizar"):
                    res_actual.monto_pesos = new_clp
                    res_actual.monto_dolates = new_usd
                    db.commit()
                    st.rerun()