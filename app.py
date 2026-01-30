import streamlit as st
import pandas as pd
import database
import yfinance as yf
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from sqlalchemy import func
import numpy as np

# Configuración de página y base de datos
st.set_page_config(page_title="Unificador de Acciones", layout="wide")
database.Base.metadata.create_all(bind=database.engine)
db = database.SessionLocal()

# --- FUNCIÓN HELPER PARA CARGAR TABLAS ---
def cargar_tabla(modelo):
    query = db.query(modelo).all()
    if not query:
        return pd.DataFrame()
    df = pd.DataFrame([u.__dict__ for u in query])
    if '_sa_instance_state' in df.columns:
        df = df.drop(columns=['_sa_instance_state'])
    return df

# --- FUNCIÓN DINÁMICA DE HISTÓRICO DIARIO CORREGIDA ---
def sincronizar_historico_diario(db):
    # 1. Encontrar la fecha de la transacción más antigua
    primera_trans = db.query(database.Transacciones_acciones).order_by(database.Transacciones_acciones.fecha).first()
    if not primera_trans:
        return 

    fecha_inicio = datetime.strptime(primera_trans.fecha, "%Y-%m-%d").date()
    hoy = datetime.now().date()
    
    # 2. Revisar la última fecha grabada
    ultima_fecha_hist = db.query(func.max(database.Historico_diario.fecha)).scalar()
    if ultima_fecha_hist:
        fecha_rastreo = datetime.strptime(ultima_fecha_hist, "%Y-%m-%d").date() + timedelta(days=1)
    else:
        fecha_rastreo = fecha_inicio

    if fecha_rastreo > hoy:
        return 

    # 3. Obtener transacciones y descargar precios
    df_trans = pd.read_sql(db.query(database.Transacciones_acciones).statement, db.bind)
    tickers = df_trans['ticker'].unique().tolist()
    
    try:
        # Descargamos los precios de cierre
        data_precios = yf.download(tickers, start=fecha_rastreo, end=hoy + timedelta(days=1))['Close']
    except Exception:
        return

    if data_precios.empty:
        return

    # 4. Reconstrucción diaria
    while fecha_rastreo < hoy:
        f_str = fecha_rastreo.strftime("%Y-%m-%d")
        t_sub = df_trans[pd.to_datetime(df_trans['fecha']).dt.date <= fecha_rastreo].copy()
        
        if not t_sub.empty:
            t_sub['neto'] = t_sub.apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1)
            saldos = t_sub.groupby(['broker', 'ticker'])['neto'].sum().reset_index()
            saldos = saldos[saldos['neto'] > 0]

            for _, fila in saldos.iterrows():
                try:
                    # EXTRACCIÓN ROBUSTA DEL PRECIO (Solución al ValueError)
                    if isinstance(data_precios, pd.DataFrame):
                        precio = data_precios.at[f_str, fila['ticker']]
                    else:
                        precio = data_precios.loc[f_str]
                    
                    # Si 'precio' sigue siendo una Serie por algún motivo, tomamos el primer valor
                    if isinstance(precio, pd.Series):
                        precio = precio.iloc[0]

                    if pd.isna(precio): 
                        continue 

                    db.add(database.Historico_diario(
                        fecha=f_str, broker=fila['broker'], ticker=fila['ticker'],
                        precio_cierre=float(precio), cantidad=float(fila['neto']), valor=float(precio * fila['neto'])
                    ))
                except Exception:
                    continue
        
        db.commit()
        fecha_rastreo += timedelta(days=1)

# Sincronización automática
if 'sync_ok' not in st.session_state:
    with st.spinner("Actualizando histórico diario..."):
        sincronizar_historico_diario(db)
    st.session_state['sync_ok'] = True

st.title("📂 Gestión de Mi Portafolio")

# --- PESTAÑAS PRINCIPALES ---
tab_acciones, tab_divisas, tab_pasivos, tab_resumen, tab_historico = st.tabs([
    "📈 Acciones", "💵 Divisas", "💰 Ingresos Pasivos", "🗓️ Resumen Mensual", "📜 Histórico Diario"
])

# 1. ACCIONES
with tab_acciones:
    st.header("Transacciones de Acciones")
    with st.expander("➕ Registrar Nueva Compra/Venta"):
        with st.form("form_acciones"):
            col1, col2, col3 = st.columns(3)
            fecha = col1.date_input("Fecha", key="acc_fecha")
            ticker = col1.text_input("Ticker").upper()
            tipo = col2.selectbox("Operación", ["compra", "venta"])
            broker = col2.selectbox("Broker", ["Racional", "Zesty", "Fintual", "Otro"])
            monto_total = col3.number_input("Monto Total (USD)", min_value=0.0)
            precio_acc = col3.number_input("Precio por Acción (USD)", min_value=0.0)
            cantidad = col3.number_input("Cantidad", min_value=0.0)
            
            if st.form_submit_button("Guardar"):
                db.add(database.Transacciones_acciones(
                    fecha=str(fecha), broker=broker, tipo_transaccion=tipo,
                    ticker=ticker, monto_total=monto_total, precio=precio_acc, cantidad=cantidad
                ))
                db.commit()
                st.success("Registrada"); st.rerun()
    
    # Actualizado: width='stretch'
    st.dataframe(cargar_tabla(database.Transacciones_acciones), width='stretch')

# 2. DIVISAS
with tab_divisas:
    st.header("Compra y Venta de Dólares")
    with st.form("form_divisas"):
        c1, c2, c3 = st.columns(3)
        f_d = c1.date_input("Fecha")
        br_d = c1.selectbox("Broker", ["Racional", "Zesty", "BancoEstado"], key="br_div")
        tipo_d = c2.selectbox("Tipo", ["compra", "venta"])
        m_clp = c2.number_input("Monto CLP")
        tc = c3.number_input("Tipo Cambio")
        cu = c3.number_input("Cantidad USD")
        if st.form_submit_button("Registrar"):
            db.add(database.Trasacciones_divisas(fecha=str(f_d), broker=br_d, tipo_transaccion=tipo_d, monto_total=m_clp, precio=tc, cantidad=cu))
            db.commit(); st.rerun()
    st.dataframe(cargar_tabla(database.Trasacciones_divisas), width='stretch')

# 3. INGRESOS PASIVOS
with tab_pasivos:
    st.header("Dividendos e Intereses")
    with st.form("form_pasivos"):
        col_a, col_b, col_c = st.columns(3)
        f_p = col_a.date_input("Fecha Pago")
        br_p = col_a.selectbox("Broker", ["Racional", "Zesty", "Fintual"])
        t_p = col_b.selectbox("Tipo", ["dividendo", "interés"])
        tk_p = col_b.text_input("Ticker/Fuente").upper()
        m_p = col_c.number_input("Monto Neto (USD)")
        if st.form_submit_button("Añadir"):
            db.add(database.Ingreso_pasivo(fecha=str(f_p), broker=br_p, tipo_ingreso=t_p, ticker=tk_p, monto=m_p))
            db.commit(); st.rerun()
    st.dataframe(cargar_tabla(database.Ingreso_pasivo), width='stretch')

# 4. RESUMEN MENSUAL
with tab_resumen:
    st.header("Resúmenes Mensuales")
    df_res = cargar_tabla(database.Resumen_mensual)
    if not df_res.empty:
        st.dataframe(df_res, width='stretch')
    with st.form("add_resumen"):
        fr = st.text_input("Mes (YYYY-MM)")
        br = st.selectbox("Broker", ["Racional", "Zesty", "Fintual"], key="br_r")
        mp = st.number_input("Monto CLP")
        md = st.number_input("Monto USD")
        if st.form_submit_button("Guardar"):
            db.add(database.Resumen_mensual(fecha=fr, broker=br, monto_pesos=mp, monto_dolares=md))
            db.commit(); st.rerun()

# 5. HISTÓRICO DIARIO
with tab_historico:
    st.header("Valor Diario del Portafolio")
    df_hist = cargar_tabla(database.Historico_diario)
    if not df_hist.empty:
        st.dataframe(df_hist.sort_values(by="fecha", ascending=False), width='stretch')
    else:
        st.info("Aún no hay datos históricos generados.")