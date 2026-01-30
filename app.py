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

# Configuración de Streamlit
st.set_page_config(page_title="Unificador de Acciones", layout="wide")
database.Base.metadata.create_all(bind=database.engine)
db = database.SessionLocal()

# --- FUNCIONES DE APOYO ---
def obtener_df(modelo):
    query = db.query(modelo).all()
    if not query: return pd.DataFrame()
    df = pd.DataFrame([u.__dict__ for u in query])
    if '_sa_instance_state' in df.columns:
        df = df.drop(columns=['_sa_instance_state'])
    return df

def mostrar_tabla_estilizada(df, tipo_tabla="acciones"):
    if df.empty: return st.info(f"Sin registros en {tipo_tabla}.")

    # Colores de texto para Brokers
    colors_br = {"Racional": "#9E9E9E", "Zesty": "#B39DDB", "Fintual": "#90CAF9", "BancoEstado": "#EF9A9A"}

    def style_df(row):
        styles = [''] * len(row)
        col_t = 'tipo_transaccion' if 'tipo_transaccion' in row else ('tipo_ingreso' if 'tipo_ingreso' in row else None)
        if col_t:
            idx = row.index.get_loc(col_t)
            styles[idx] = 'color: #2e7d32; font-weight: bold;' if row[col_t] in ['compra', 'dividendo', 'interés'] else 'color: #d32f2f; font-weight: bold;'
        if 'broker' in row:
            idx_b = row.index.get_loc('broker')
            styles[idx_b] = f'color: {colors_br.get(row["broker"], "")}; font-weight: bold;'
        return styles

    configs = {
        "acciones": {"order": ["fecha", "broker", "tipo_transaccion", "ticker", "monto_total", "precio", "cantidad"],
                     "config": {"monto_total": st.column_config.NumberColumn("Monto USD", format="$ %.2f"), "id": None, "cantidad": st.column_config.NumberColumn("Cant.", format="%.8f")}},
        "divisas": {"order": ["fecha", "broker", "tipo_transaccion", "monto_total", "precio", "cantidad"],
                    "config": {"monto_total": st.column_config.NumberColumn("Monto CLP", format="$ %.0f"), "id": None}},
        "pasivos": {"order": ["fecha", "broker", "tipo_ingreso", "ticker", "monto"],
                    "config": {"monto": st.column_config.NumberColumn("Monto USD", format="$ %.2f"), "id": None}},
        "resumen_mensual": {"order": ["fecha", "broker", "monto_pesos", "monto_dolares"],
                    "config": {"monto_pesos": st.column_config.NumberColumn("Pesos (CLP)", format="$ %.0f"), "monto_dolares": st.column_config.NumberColumn("Dólares (USD)", format="$ %.2f"), "id": None}},
        "historico": {"order": ["fecha", "broker", "ticker", "precio_cierre", "monto_total", "cantidad", "valor"],
                    "config": {"precio_cierre": st.column_config.NumberColumn("Cierre", format="$ %.2f"),"monto_total": st.column_config.NumberColumn("Invertido", format="$ %.2f"),"cantidad": st.column_config.NumberColumn("Cantidad", format="%.8f"),"valor": st.column_config.NumberColumn("Valor Portafolio", format="$ %.2f"),"id": None
            }
        }
    }

    cfg = configs.get(tipo_tabla, {"order": list(df.columns), "config": {"id": None}})
    st.dataframe(df.style.apply(style_df, axis=1), column_order=cfg["order"], column_config=cfg["config"], width="stretch", height="content")

# --- SINCRONIZACIÓN (HASTA AYER / DÍAS DE BOLSA) ---
def sincronizar_todo(db):
    # Límite estricto: Ayer
    ayer = datetime.now().date() - timedelta(days=1)
    df_acc = obtener_df(database.Transacciones_acciones)
    if df_acc.empty: return st.warning("Añade transacciones primero.")

    fecha_inicio = pd.to_datetime(df_acc['fecha']).min().date()
    tickers = df_acc['ticker'].unique().tolist()
    
    # Descarga solo hasta ayer
    data = yf.download(tickers, start=fecha_inicio, end=ayer + timedelta(days=1))['Close']
    data_usdclp = yf.download("CLP=X", start=fecha_inicio, end=ayer + timedelta(days=1))['Close']
    
    if data.empty or data_usdclp.empty: return

    db.query(database.Historico_diario).delete()
    db.query(database.Resumen_cartera_diaria).delete()
    
    df_div = obtener_df(database.Trasacciones_divisas)
    df_pas = obtener_df(database.Ingreso_pasivo)

    for fecha_bolsa in data.index:
        f_str = fecha_bolsa.strftime("%Y-%m-%d")
        f_dt = fecha_bolsa.date()
        
        # --- SOLUCIÓN ERROR AMBIGÜEDAD (Tipo de Cambio) ---
        try:
            val_d = data_usdclp.loc[fecha_bolsa]
        except KeyError:
            val_d = data_usdclp.asof(fecha_bolsa)
        
        # Forzar a valor único si es una Serie
        cambio_actual = val_d.iloc[0] if hasattr(val_d, 'iloc') else val_d
        if pd.isna(cambio_actual): continue

        # --- PROCESAMIENTO DE ACCIONES ---
        t_hoy = df_acc[pd.to_datetime(df_acc['fecha']).dt.date <= f_dt].copy()
        t_hoy['n_cant'] = t_hoy.apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1)
        t_hoy['n_monto'] = t_hoy.apply(lambda x: x['monto_total'] if x['tipo_transaccion'] == 'compra' else -x['monto_total'], axis=1)
        
        saldos = t_hoy.groupby(['broker', 'ticker']).agg({'n_cant': 'sum', 'n_monto': 'sum'}).reset_index()
        saldos = saldos[saldos['n_cant'] > 1e-8]

        for _, row in saldos.iterrows():
            try:
                # --- SOLUCIÓN ERROR AMBIGÜEDAD (Precios Acciones) ---
                p_raw = data.loc[fecha_bolsa, row['ticker']] if len(tickers) > 1 else data.loc[fecha_bolsa]
                p = p_raw.iloc[0] if hasattr(p_raw, 'iloc') else p_raw
                if pd.isna(p): continue
                
                db.add(database.Historico_diario(
                    fecha=f_str, broker=row['broker'], ticker=row['ticker'], precio_cierre=float(p),
                    monto_total=float(row['n_monto']), cantidad=float(row['n_cant']), valor=float(p * row['n_cant'])
                ))
            except: continue

        # --- MÉTRICAS DE CARTERA (USD y CLP) ---
        for br in df_acc['broker'].unique():
            df_div_br = df_div[(df_div['broker'] == br) & (pd.to_datetime(df_div['fecha']).dt.date <= f_dt)]
            iny_usd = df_div_br.apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1).sum()
            iny_clp = df_div_br.apply(lambda x: x['monto_total'] if x['tipo_transaccion'] == 'compra' else -x['monto_total'], axis=1).sum()
            
            flux_usd = t_hoy[t_hoy['broker'] == br]['n_monto'].sum() * -1
            divs_usd = df_pas[(df_pas['broker'] == br) & (pd.to_datetime(df_pas['fecha']).dt.date <= f_dt)]['monto'].sum()
            
            caja_usd = iny_usd + flux_usd + divs_usd
            # Valor acciones USD
            v_acc_usd = saldos[saldos['broker'] == br].apply(
                lambda x: x['n_cant'] * (data.loc[fecha_bolsa, x['ticker']].iloc[0] if hasattr(data.loc[fecha_bolsa, x['ticker']], 'iloc') else data.loc[fecha_bolsa, x['ticker']]), axis=1).sum()
            
            total_usd = v_acc_usd + caja_usd
            total_clp = total_usd * cambio_actual
            retorno_clp = total_clp - iny_clp
            
            db.add(database.Resumen_cartera_diaria(
                fecha=f_str, broker=br, valor_acciones=float(v_acc_usd), caja=float(caja_usd),
                total=float(total_usd), total_pesos=float(total_clp),
                capital_invertido=float(iny_usd), capital_invertido_pesos=float(iny_clp),
                retorno=float(total_usd - iny_usd), retorno_pesos=float(retorno_clp),
                cambio_dolar=float(cambio_actual)
            ))
    db.commit()
    st.success("Sincronización histórica (hasta ayer) completada.")

# --- INTERFAZ ---
st.title("📂 Mi Portafolio de Inversiones")
tab_graficos, tab_acciones, tab_divisas, tab_pasivos, tab_mensual = st.tabs([
    "📈 Gráficos e Histórico", "📊 Acciones", "💵 Divisas", "💰 Ingresos Pasivos", "🗓️ Resumen Mensual"
])

# 1. PESTAÑA GRÁFICOS (ARRIBA DEL HISTÓRICO)
with tab_graficos:
    st.header("Análisis de Rendimiento")
    if st.button("🔄 Sincronizar (Hasta Ayer)"):
        with st.spinner("Procesando datos..."): sincronizar_todo(db)
        st.rerun()

    df_r = obtener_df(database.Resumen_cartera_diaria)
    df_h = obtener_df(database.Historico_diario)

    if not df_r.empty:
        # --- FILTROS ---
        c1, c2, c3, c4 = st.columns([1, 1, 2, 0.5])
        br_sel = c1.multiselect("Bróker", df_r['broker'].unique(), default=df_r['broker'].unique())
        met_sel = c2.multiselect("Métricas Cartera", ["Total", "Retorno", "Caja", "Capital Invertido"], default=["Retorno"])
        tk_sel = c3.multiselect("Filtrar Acción (Selecciona para aislar)", df_h['ticker'].unique())
        with c4:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            ver_acciones = st.checkbox("Ver Acciones", value=True)

        # Gráfico
        fig = go.Figure()
        df_p = df_r[df_r['broker'].isin(br_sel)].groupby('fecha').sum(numeric_only=True).reset_index()
        
        # Métricas principales (Líneas delgadas width=1.5)
        if "Total" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['total'], name="TOTAL", line=dict(color='white', width=1.5)))
        if "Retorno" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['retorno'], name="RETORNO", line=dict(color='#00e676', width=1.5)))
        if "Caja" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['caja'], name="CAJA", line=dict(dash='dash', color='#ff9100', width=1.5)))
        if "Capital Invertido" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['capital_invertido'], name="CAP. INVERTIDO", line=dict(color='gray', dash='dot', width=1.5)))

        # --- LÓGICA DE ACCIONES ---
        if ver_acciones:
            # Si no hay selección, mostramos todos los tickers disponibles
            tickers_a_mostrar = tk_sel if tk_sel else df_h['ticker'].unique()
            for tk in tickers_a_mostrar:
                # Filtrar historial por ticker y bróker
                df_tk = df_h[(df_h['ticker'] == tk) & (df_h['broker'].isin(br_sel))].groupby('fecha').sum(numeric_only=True).reset_index()
                df_tk = pd.merge(df_p[['fecha']], df_tk, on='fecha', how='left')
                
                # 1. Línea de Valor Actual (Sólida)
                fig.add_trace(go.Scatter(
                    x=df_tk['fecha'], 
                    y=df_tk['valor'], 
                    name=f"Valor: {tk}", 
                    line=dict(width=1), 
                    connectgaps=False
                ))

                # 2. Línea de Monto Invertido (Punteada - Solo si el ticker fue seleccionado explícitamente)
                if tk_sel:
                    fig.add_trace(go.Scatter(
                        x=df_tk['fecha'], 
                        y=df_tk['monto_total'], 
                        name=f"Invertido: {tk}", 
                        line=dict(width=1, dash='dot'), 
                        opacity=0.6,
                        connectgaps=False
                    ))

        # --- DISEÑO Y BOTONES ---
        fig.update_xaxes(rangeselector=dict(buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all", label="Máx")
        ]), bgcolor="#1E1E1E", activecolor="#2E7D32", y=1.05))

        fig.update_layout(
            template="plotly_dark", height=650, hovermode="x unified",
            yaxis=dict(tickformat=".2f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1)
        )
        fig.update_traces(hovertemplate='%{y:.2f}')
        st.plotly_chart(fig, width="stretch")

        st.subheader("📜 Detalle Diario")
        mostrar_tabla_estilizada(df_h.sort_values(by="fecha", ascending=False), "historico")

# 2. ACCIONES
with tab_acciones:
    st.header("Transacciones de Acciones")
    with st.form("form_acc"):
        col1, col2, col3 = st.columns(3)
        f = col1.date_input("Fecha"); t = col1.text_input("Ticker").upper()
        tipo = col2.selectbox("Operación", ["compra", "venta"]); br = col2.selectbox("Broker", ["Racional", "Zesty"])
        mt = col3.number_input("Monto Total (USD)"); pr = col3.number_input("Precio Acción"); ct = col3.number_input("Cantidad", format="%.8f")
        if st.form_submit_button("Guardar"):
            db.add(database.Transacciones_acciones(fecha=str(f), broker=br, tipo_transaccion=tipo, ticker=t, monto_total=mt, precio=pr, cantidad=ct))
            db.commit(); st.rerun()
    mostrar_tabla_estilizada(obtener_df(database.Transacciones_acciones), "acciones")

# 3. DIVISAS
with tab_divisas:
    st.header("Historial de Divisas")
    with st.form("f_div"):
        c1, c2, c3 = st.columns(3)
        fd = c1.date_input("Fecha"); bd = c1.selectbox("Broker", ["Racional", "Zesty"])
        td = c2.selectbox("Tipo", ["compra", "venta"]); mc = c2.number_input("Monto CLP")
        tc = c3.number_input("Tipo Cambio"); cu = c3.number_input("Cantidad USD")
        if st.form_submit_button("Registrar"):
            db.add(database.Trasacciones_divisas(fecha=str(fd), broker=bd, tipo_transaccion=td, monto_total=mc, precio=tc, cantidad=cu))
            db.commit(); st.rerun()
    mostrar_tabla_estilizada(obtener_df(database.Trasacciones_divisas), "divisas")

# 4. INGRESOS PASIVOS
with tab_pasivos:
    st.header("Dividendos e Intereses")
    with st.form("f_pas"):
        c1, c2, c3 = st.columns(3)
        fp = c1.date_input("Fecha"); bp = c1.selectbox("Broker", ["Racional", "Zesty"])
        tp = c2.selectbox("Tipo", ["dividendo", "interés"]); tk = c2.text_input("Ticker").upper()
        m = c3.number_input("Monto USD")
        if st.form_submit_button("Añadir"):
            db.add(database.Ingreso_pasivo(fecha=str(fp), broker=bp, tipo_ingreso=tp, ticker=tk, monto=m))
            db.commit(); st.rerun()
    mostrar_tabla_estilizada(obtener_df(database.Ingreso_pasivo), "pasivos")

# 5. RESUMEN MENSUAL
with tab_mensual:
    st.header("Cierre de Mes Oficial")
    with st.form("form_mes"):
        col1, col2 = st.columns(2)
        fm = col1.text_input("Mes (YYYY-MM)"); bm = col1.selectbox("Broker", ["Racional", "Zesty"])
        mp = col2.number_input("Monto Pesos (CLP)"); md = col2.number_input("Monto Dólares (USD)")
        if st.form_submit_button("Guardar Cierre Mensual"):
            db.add(database.Resumen_mensual(fecha=fm, broker=bm, monto_pesos=mp, monto_dolares=md))
            db.commit(); st.rerun()
    mostrar_tabla_estilizada(obtener_df(database.Resumen_mensual), "resumen_mensual")