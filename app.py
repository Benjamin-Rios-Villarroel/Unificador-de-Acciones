import streamlit as st
import pandas as pd
import database
import yfinance as yf
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from sqlalchemy import func
import numpy as np
import pytz
import plotly.graph_objects as go

# Configuración de página y base de datos
st.set_page_config(page_title="Unificador de Acciones", layout="wide")
database.Base.metadata.create_all(bind=database.engine)
db = database.SessionLocal()

# --- FUNCIÓN HELPER PARA CARGAR TABLAS ---
def mostrar_tabla_estilizada(df, tipo_tabla="acciones"):
    if df.empty:
        return st.info(f"No hay datos registrados en {tipo_tabla} aún.")

    # 1. Definir Colores de Texto para Brokers
    broker_text_colors = {
        "Racional": "#9E9E9E",   # Gris claro
        "Zesty": "#B39DDB",      # Morado claro
        "Fintual": "#90CAF9",    # Azul claro
        "BancoEstado": "#EF9A9A" # Rojo claro (opcional)
    }

    # 2. Lógica de Estilo (Letras)
    def style_df(row):
        styles = [''] * len(row)
        
        # Color para Tipo de Transacción o Ingreso
        col_tipo = None
        if 'tipo_transaccion' in row: col_tipo = 'tipo_transaccion'
        elif 'tipo_ingreso' in row: col_tipo = 'tipo_ingreso'
        
        if col_tipo:
            idx_t = row.index.get_loc(col_tipo)
            if row[col_tipo] in ['compra', 'dividendo', 'interés']:
                styles[idx_t] = 'color: #2e7d32; font-weight: bold;'
            elif row[col_tipo] == 'venta':
                styles[idx_t] = 'color: #d32f2f; font-weight: bold;'
        
        # Color de letras para Broker
        if 'broker' in row:
            idx_b = row.index.get_loc('broker')
            color = broker_text_colors.get(row['broker'], '')
            if color:
                styles[idx_b] = f'color: {color}; font-weight: bold;'
        return styles

    # 3. Configuración de Columnas
    configs = {
        "acciones": {
            "order": ["fecha", "broker", "tipo_transaccion", "ticker", "monto_total", "precio", "cantidad"],
            "config": {
                "monto_total": st.column_config.NumberColumn("Monto Total", format="$ %.2f"),
                "precio": st.column_config.NumberColumn("Precio Acción", format="$ %.2f"),
                "cantidad": st.column_config.NumberColumn("Cantidad", format="%.8f"),
                "id": None
            }
        },
        "divisas": {
            "order": ["fecha", "broker", "tipo_transaccion", "monto_total", "precio", "cantidad"],
            "config": {
                "monto_total": st.column_config.NumberColumn("Monto CLP", format="$ %.0f"),
                "precio": st.column_config.NumberColumn("Tipo Cambio", format="$ %.2f"),
                "cantidad": st.column_config.NumberColumn("Total USD", format="$ %.2f"),
                "id": None
            }
        },
        "pasivos": {
            "order": ["fecha", "broker", "tipo_ingreso", "ticker", "monto"],
            "config": {
                "monto": st.column_config.NumberColumn("Monto USD", format="$ %.2f"),
                "id": None
            }
        },
        "resumen": {
            "order": ["fecha", "broker", "monto_pesos", "monto_dolares"],
            "config": {
                "monto_pesos": st.column_config.NumberColumn("Pesos (CLP)", format="$ %.0f"),
                "monto_dolares": st.column_config.NumberColumn("Dólares (USD)", format="$ %.2f"),
                "id": None
            }
        },
        "historico": {
            "order": ["fecha", "broker", "ticker", "precio_cierre", "cantidad", "valor"],
            "config": {
                "precio_cierre": st.column_config.NumberColumn("Cierre", format="$ %.2f"),
                "cantidad": st.column_config.NumberColumn("Cantidad", format="%.8f"),
                "valor": st.column_config.NumberColumn("Valor Portafolio", format="$ %.2f"),
                "id": None
            }
        }
    }

    # Aseguramos que el orden sea una lista de Python para evitar el ValueError
    cfg = configs.get(tipo_tabla)
    column_order = cfg["order"] if cfg else None
    column_config = cfg["config"] if cfg else {"id": None}

    st.dataframe(
        df.style.apply(style_df, axis=1),
        column_order=column_order,
        column_config=column_config,
        width="stretch",
        height="content"
    )

# --- FUNCIÓN DE CARGA ---
def obtener_df(modelo):
    query = db.query(modelo).all()
    if not query: return pd.DataFrame()
    df = pd.DataFrame([u.__dict__ for u in query])
    return df.drop(columns=['_sa_instance_state']) if '_sa_instance_state' in df.columns else df

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
            cantidad = col3.number_input("Cantidad", min_value=0.0, format="%.8f")
            
            if st.form_submit_button("Guardar"):
                db.add(database.Transacciones_acciones(
                    fecha=str(fecha), broker=broker, tipo_transaccion=tipo,
                    ticker=ticker, monto_total=monto_total, precio=precio_acc, cantidad=cantidad
                ))
                db.commit()
                del st.session_state['sync_ok'] # Forzar resincronización
                st.success("Registrada"); st.rerun()
    
    mostrar_tabla_estilizada(obtener_df(database.Transacciones_acciones), "acciones")

# 2. DIVISAS
with tab_divisas:
    st.header("Compra y Venta de Dólares")
    with st.expander("➕ Registrar Divisa"):
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
    mostrar_tabla_estilizada(obtener_df(database.Trasacciones_divisas), "divisas")

# 3. INGRESOS PASIVOS (Corregido)
with tab_pasivos:
    st.header("Dividendos e Intereses")
    with st.expander("➕ Registrar Ingreso"):
        with st.form("form_pasivos"):
            col_a, col_b, col_c = st.columns(3)
            f_p = col_a.date_input("Fecha Pago")
            br_p = col_a.selectbox("Broker", ["Racional", "Zesty", "Fintual"])
            t_p = col_b.selectbox("Tipo", ["dividendo", "interés"])
            tk_p = col_b.text_input("Ticker/Fuente").upper()
            m_p = col_c.number_input("Monto Neto (USD)")
            if st.form_submit_button("Añadir"):
                db.add(database.Ingreso_pasivo(fecha=str(f_p), broker=br_p, tipo_ingreso=t_p, ticker=tk_p, monto=m_p))
                db.commit()
                st.rerun()
    mostrar_tabla_estilizada(obtener_df(database.Ingreso_pasivo), "pasivos")

# 4. RESUMEN MENSUAL (Corregido)
with tab_resumen:
    st.header("Resúmenes Mensuales")
    mostrar_tabla_estilizada(obtener_df(database.Resumen_mensual), "resumen")
    with st.form("add_resumen"):
        fr = st.text_input("Mes (YYYY-MM)")
        br = st.selectbox("Broker", ["Racional", "Zesty", "Fintual"], key="br_res_form")
        mp = st.number_input("Monto CLP")
        md = st.number_input("Monto USD")
        if st.form_submit_button("Guardar Resumen"):
            db.add(database.Resumen_mensual(fecha=fr, broker=br, monto_pesos=mp, monto_dolares=md))
            db.commit()
            st.rerun()

# 5. HISTÓRICO DIARIO (Corregido)
with tab_historico:
    st.header("Valor Diario del Portafolio")
    mostrar_tabla_estilizada(obtener_df(database.Historico_diario), "historico")

# --- FUNCIÓN PARA CALCULAR MÉTRICAS DIARIAS ---
def sincronizar_resumen_cartera(db):
    # 1. Cargar todos los datos necesarios
    df_acc = obtener_df(database.Transacciones_acciones)
    df_div = obtener_df(database.Trasacciones_divisas)
    df_pas = obtener_df(database.Ingreso_pasivo)
    df_hist = obtener_df(database.Historico_diario) # Esta debe estar llena primero
    
    # Validar que existan datos mínimos para empezar
    if df_acc.empty and df_div.empty:
        st.warning("No hay transacciones suficientes para generar el gráfico.")
        return

    # Limpiar tabla para evitar duplicados al recalcular
    db.query(database.Resumen_cartera_diaria).delete()
    db.commit()
    
    # 2. Encontrar la fecha de la primera transacción histórica
    fechas_posibles = []
    if not df_acc.empty: fechas_posibles.append(pd.to_datetime(df_acc['fecha']).min())
    if not df_div.empty: fechas_posibles.append(pd.to_datetime(df_div['fecha']).min())
    
    fecha_inicio = min(fechas_posibles).date()
    hoy = datetime.now().date()
    
    # Obtener lista de todos los brokers involucrados
    brokers = pd.concat([df_acc['broker'], df_div['broker']]).unique().tolist()
    
    # 3. Bucle diario (Desde el inicio hasta hoy)
    for fecha_actual in pd.date_range(fecha_inicio, hoy):
        f_str = fecha_actual.strftime("%Y-%m-%d")
        
        for br in brokers:
            # --- CÁLCULO DE CAJA (Efectivo disponible en el bróker) ---
            # Dinero metido/sacado vía divisas
            div_f = df_div[(df_div['broker'] == br) & (pd.to_datetime(df_div['fecha']) <= fecha_actual)]
            usd_inyectado = div_f.apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1).sum()
            
            # Dinero movido por compra/venta de acciones
            acc_f = df_acc[(df_acc['broker'] == br) & (pd.to_datetime(df_acc['fecha']) <= fecha_actual)]
            usd_flujo_acc = acc_f.apply(lambda x: x['monto_total'] if x['tipo_transaccion'] == 'venta' else -x['monto_total'], axis=1).sum()
            
            # Dinero ganado por dividendos/intereses
            pas_f = df_pas[(df_pas['broker'] == br) & (pd.to_datetime(df_pas['fecha']) <= fecha_actual)]
            usd_pasivos = pas_f['monto'].sum() if not pas_f.empty else 0
            
            caja_dia = usd_inyectado + usd_flujo_acc + usd_pasivos
            
            # --- VALOR DE LAS ACCIONES ESE DÍA ---
            val_acc_dia = df_hist[(df_hist['broker'] == br) & (df_hist['fecha'] == f_str)]['valor'].sum()
            
            # --- TOTAL Y RETORNO ---
            total_dia = val_acc_dia + caja_dia
            retorno_dia = total_dia - usd_inyectado # Ganancia neta sobre capital aportado
            
            # Guardar en la base de datos
            nueva_metrica = database.Resumen_cartera_diaria(
                fecha=f_str,
                broker=br,
                valor_acciones=float(val_acc_dia),
                caja=float(caja_dia),
                total=float(total_dia),
                capital_invertido=float(usd_inyectado),
                retorno=float(retorno_dia)
            )
            db.add(nueva_metrica)
            
    db.commit()
    st.success(f"Métricas calculadas desde {fecha_inicio} hasta hoy.")

# --- PESTAÑA DE GRÁFICOS ---
with tab_historico: # O puedes crear una nueva tab_graficos
    st.header("📈 Análisis de Rendimiento")
    
    if st.button("🔄 Recalcular Métricas"):
        sincronizar_resumen_cartera(db)
        st.rerun()

    df_r = obtener_df(database.Resumen_cartera_diaria)
    df_h = obtener_df(database.Historico_diario)

    if not df_r.empty:
        # Filtros Multiselect
        c1, c2 = st.columns(2)
        br_sel = c1.multiselect("Brokers", df_r['broker'].unique(), default=df_r['broker'].unique())
        metricas = c2.multiselect("Mostrar Métricas", ["Total", "Retorno", "Caja"], default=["Total", "Retorno"])
        
        df_p = df_r[df_r['broker'].isin(br_sel)].groupby('fecha').sum(numeric_only=True).reset_index()
        
        fig = go.Figure()
        
        if "Total" in metricas:
            fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['total'], name="Total Cartera", line=dict(color='white', width=3)))
        if "Retorno" in metricas:
            fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['retorno'], name="Retorno ($)", line=dict(color='#00e676')))
        if "Caja" in metricas:
            fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['caja'], name="Caja (Efectivo)", line=dict(dash='dash')))

        # Agregar líneas de acciones individuales
        if st.checkbox("Ver detalle por Tickers"):
            for tk in df_h['ticker'].unique():
                df_tk = df_h[(df_h['ticker'] == tk) & (df_h['broker'].isin(br_sel))].groupby('fecha').sum(numeric_only=True).reset_index()
                fig.add_trace(go.Scatter(x=df_tk['fecha'], y=df_tk['valor'], name=tk, mode='lines', opacity=0.6))

        fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")