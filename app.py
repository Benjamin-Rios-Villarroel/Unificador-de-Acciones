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
                     "config": {"monto_total": st.column_config.NumberColumn("Monto USD", format="$ %.2f"), "precio": st.column_config.NumberColumn("Precio USD", format="$ %.2f"), "id": None, "cantidad": st.column_config.NumberColumn("Cant.", format="%.8f")}},
        "divisas": {"order": ["fecha", "broker", "tipo_transaccion", "monto_total", "precio", "cantidad"],
                    "config": {"monto_total": st.column_config.NumberColumn("Monto CLP", format="$ %.0f"), "precio": st.column_config.NumberColumn("Cambio CLP", format="$ %.2f"), "cantidad": st.column_config.NumberColumn("Cantidad USD", format="$ %.2f"), "id": None}},
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

    # --- NUEVA LÓGICA: Calcular Costo de Posición (Weighted Average) ---
    # Ordenamos por fecha para procesar cronológicamente
    df_acc = df_acc.sort_values(by="fecha")
    
    # Calculamos el impacto en el costo para cada transacción
    cost_impacts = []
    cash_flows = []
    saldos_temp = {} # { (broker, ticker): [qty, cost_basis] }

    for _, row in df_acc.iterrows():
        key = (row['broker'], row['ticker'])
        if key not in saldos_temp:
            saldos_temp[key] = [0.0, 0.0]
        
        qty_actual, cost_actual = saldos_temp[key]
        
        if row['tipo_transaccion'] == 'compra':
            impacto = row['monto_total']
            saldos_temp[key][0] += row['cantidad']
            saldos_temp[key][1] += impacto
            cash_flows.append(-row['monto_total']) # Dinero que sale de la caja
        else: # venta
            # El impacto en el costo es proporcional a lo que se vende del costo base actual
            if qty_actual > 0:
                proporcion_vendida = row['cantidad'] / qty_actual
                impacto = -(cost_actual * proporcion_vendida)
                saldos_temp[key][0] -= row['cantidad']
                saldos_temp[key][1] += impacto
            else:
                impacto = 0
            
            # Si la cantidad queda en cero, reseteamos el costo a cero absoluto
            if saldos_temp[key][0] < 1e-8:
                saldos_temp[key] = [0.0, 0.0]
                
            cash_flows.append(row['monto_total']) # Dinero que entra a la caja
            
        cost_impacts.append(impacto)

    df_acc['cost_impact'] = cost_impacts
    df_acc['cash_flow'] = cash_flows

    # --- CONTINUACIÓN DE LA SINCRONIZACIÓN ---
    fecha_inicio = pd.to_datetime(df_acc['fecha']).min().date()
    tickers = df_acc['ticker'].unique().tolist()
    
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
        
        # Tipo de cambio con corrección de ambigüedad
        try:
            val_d = data_usdclp.loc[fecha_bolsa]
        except KeyError:
            val_d = data_usdclp.asof(fecha_bolsa)
        cambio_actual = val_d.iloc[0] if hasattr(val_d, 'iloc') else val_d
        if pd.isna(cambio_actual): continue

        # Transacciones hasta hoy
        t_hoy = df_acc[pd.to_datetime(df_acc['fecha']).dt.date <= f_dt].copy()
        
        # Cantidad neta para saber qué acciones hay en cartera
        t_hoy['n_cant'] = t_hoy.apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1)
        
        # Agrupamos por broker/ticker
        # n_monto ahora suma los impactos de costo (se resetea al vender y suma al comprar)
        saldos = t_hoy.groupby(['broker', 'ticker']).agg({
            'n_cant': 'sum',
            'cost_impact': 'sum'
        }).reset_index()
        
        saldos = saldos[saldos['n_cant'] > 1e-8]

        for _, row in saldos.iterrows():
            try:
                p_raw = data.loc[fecha_bolsa, row['ticker']] if len(tickers) > 1 else data.loc[fecha_bolsa]
                p = p_raw.iloc[0] if hasattr(p_raw, 'iloc') else p_raw
                if pd.isna(p): continue
                
                db.add(database.Historico_diario(
                    fecha=f_str, broker=row['broker'], ticker=row['ticker'], precio_cierre=float(p),
                    monto_total=float(row['cost_impact']), # COSTO DE LA POSICIÓN ACTUAL
                    cantidad=float(row['n_cant']), 
                    valor=float(p * row['n_cant'])
                ))
            except: continue

        # --- MÉTRICAS DE CARTERA ---
        for br in df_acc['broker'].unique():
            df_div_br = df_div[(df_div['broker'] == br) & (pd.to_datetime(df_div['fecha']).dt.date <= f_dt)]
            iny_usd = df_div_br.apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1).sum()
            iny_clp = df_div_br.apply(lambda x: x['monto_total'] if x['tipo_transaccion'] == 'compra' else -x['monto_total'], axis=1).sum()
            
            # La caja usa el flujo de efectivo real (compras restan, ventas suman)
            flux_usd = t_hoy[t_hoy['broker'] == br]['cash_flow'].sum()
            divs_usd = df_pas[(df_pas['broker'] == br) & (pd.to_datetime(df_pas['fecha']).dt.date <= f_dt)]['monto'].sum()
            
            caja_usd = iny_usd + flux_usd + divs_usd
            
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
    st.success("Sincronización completada con corrección de costo de posición.")

# --- INTERFAZ ---
st.title("📂 Mi Portafolio de Inversiones")
tab_acciones, tab_graficos, tab_divisas, tab_pasivos = st.tabs([
    "📊 Portafolio", "📈 Histórico", "💵 Valores Histórico", "💰 Ingresos Pasivos"
])

# 1. PESTAÑA GRÁFICOS 
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

        # Preparación de datos del gráfico
        df_p = df_r[df_r['broker'].isin(br_sel)].groupby('fecha').sum(numeric_only=True).reset_index()

        # --- RESUMEN DE VALORES CENTRADOS CON ESPACIO REDUCIDO ---
        if not df_p.empty:
            ultimo_total = df_p['total'].iloc[-1]
            ultimo_retorno = df_p['retorno'].iloc[-1]
            color_retorno = "#00e676" if ultimo_retorno >= 0 else "#ff5252"
        
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: -25px;">
                    <p style="font-size: 50px; font-weight: bold; margin-bottom: -20px; color: white;">
                        ${ultimo_total:,.2f}
                    </p>
                    <p style="font-size: 25px; font-weight: 500; color: {color_retorno}; margin-top: 0px;">
                        ${ultimo_retorno:,.2f} USD
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Gráfico
        fig = go.Figure()
        
        # Lógica de Rango Temporal (1 Año)
        ultima_f = pd.to_datetime(df_p['fecha']).max()
        inicio_f = ultima_f - pd.DateOffset(years=1)
        primera_f = pd.to_datetime(df_p['fecha']).min()
        if inicio_f < primera_f:
            inicio_f = primera_f

        # Métricas principales
        if "Total" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['total'], name="TOTAL", line=dict(color='white', width=1.5)))
        if "Retorno" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['retorno'], name="RETORNO", line=dict(color='#00e676', width=1.5)))
        if "Caja" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['caja'], name="CAJA", line=dict(dash='dash', color='#ff9100', width=1.5)))
        if "Capital Invertido" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['capital_invertido'], name="CAP. INVERTIDO", line=dict(color='gray', dash='dot', width=1.5)))

        # Lógica de Acciones
        if ver_acciones:
            tickers_a_mostrar = tk_sel if tk_sel else df_h['ticker'].unique()
            for tk in tickers_a_mostrar:
                df_tk = df_h[(df_h['ticker'] == tk) & (df_h['broker'].isin(br_sel))].groupby('fecha').sum(numeric_only=True).reset_index()
                df_tk = pd.merge(df_p[['fecha']], df_tk, on='fecha', how='left')
                
                fig.add_trace(go.Scatter(
                    x=df_tk['fecha'], y=df_tk['valor'], 
                    name=f"Valor: {tk}", line=dict(width=1), connectgaps=False
                ))

                if tk_sel:
                    fig.add_trace(go.Scatter(
                        x=df_tk['fecha'], y=df_tk['monto_total'], 
                        name=f"Inv: {tk}", line=dict(width=1, dash='dot'), 
                        opacity=0.6, connectgaps=False
                    ))

        # --- DISEÑO OPTIMIZADO: LEYENDA ABAJO Y MÁRGENES REDUCIDOS ---
        fig.update_xaxes(
            range=[inicio_f, ultima_f],
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="Máx")
                ]), 
                bgcolor="#1E1E1E", 
                activecolor="#2E7D32", 
                y=1.01 # Bajamos el selector para que esté más cerca del eje
            )
        )

        fig.update_layout(
            template="plotly_dark", 
            height=550, 
            hovermode="x unified",
            margin=dict(t=10, b=80, l=10, r=10), # Reducimos margen superior a 10px
            yaxis=dict(tickformat=".2f"),
            # Leyenda abajo y centrada
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        fig.update_traces(hovertemplate='%{y:.2f}')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📜 Detalle Diario")
        mostrar_tabla_estilizada(df_h.sort_values(by="fecha", ascending=False), "historico")

# 2. ACCIONES
with tab_acciones:
    st.header("📊 Resumen de Tenencias (Tiempo Real)")
    
    df_h = obtener_df(database.Historico_diario)
    df_r = obtener_df(database.Resumen_cartera_diaria)
    
    if not df_h.empty and not df_r.empty:
        # 1. Datos base del último cierre
        ultima_fecha = df_h['fecha'].max()
        df_h_hoy = df_h[df_h['fecha'] == ultima_fecha].copy()
        df_r_hoy = df_r[df_r['fecha'] == ultima_fecha].copy()
        
        # 2. Consolidar stocks
        port = df_h_hoy.groupby('ticker').agg({
            'cantidad': 'sum',
            'monto_total': 'sum'
        }).reset_index()
        
        # --- LÓGICA TIEMPO REAL ---
        tickers = port['ticker'].unique().tolist()
        with st.spinner("Actualizando precios en vivo..."):
            try:
                data_live = yf.download(tickers, period="1d", interval="1m")['Close'].iloc[-1]
                if len(tickers) > 1:
                    port['Cotización'] = port['ticker'].map(data_live)
                else:
                    port['Cotización'] = data_live
            except:
                st.warning("Precios en vivo no disponibles. Usando último cierre.")
                port['Cotización'] = port['ticker'].map(df_h_hoy.groupby('ticker')['precio_cierre'].mean())

        # 3. Cálculos de activos
        port['Valor Actual'] = port['cantidad'] * port['Cotización']
        port['P. Promedio'] = port['monto_total'] / port['cantidad']
        port['Ganancia'] = port['Valor Actual'] - port['monto_total']
        port['% Ganancia'] = (port['Ganancia'] / port['monto_total']) * 100
        
        # 4. Cálculos para los apartados superiores
        total_efectivo = df_r_hoy['caja'].sum()
        total_acciones_val = port['Valor Actual'].sum()
        total_acciones_inv = port['monto_total'].sum()
        total_acciones_gan = port['Ganancia'].sum()
        
        # Total actual de toda la cartera (Valor de mercado + Efectivo)
        total_mercado = total_acciones_val + total_efectivo
        # Capital invertido actual (Depósitos totales)
        total_deposito = df_r_hoy['capital_invertido'].sum()
        
        # --- NUEVA LÓGICA DE ALH (ATH) BASADA EN RETORNO ---
        # 1. Encontramos el máximo retorno histórico
        max_retorno_hist = df_r['retorno'].max()
        
        # 2. El ATH Ajustado es: Capital actual + el mejor retorno que hemos tenido
        # Si el retorno actual en vivo es mayor al histórico, usamos el actual
        retorno_actual_vivo = total_mercado - total_deposito
        mejor_retorno = max(max_retorno_hist, retorno_actual_vivo)
        
        total_ath_ajustado = total_deposito + mejor_retorno
        diferencia_ath = total_mercado - total_ath_ajustado
        
        porc_caja = (total_efectivo / total_mercado) * 100 if total_mercado != 0 else 0

        # --- APARTADOS DE RESUMEN SUPERIOR (3 COLUMNAS) ---
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9100;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Efectivo en Caja</p>
                    <h3 style="margin: 0; color: white;">$ {total_efectivo:,.2f} USD</h3>
                    <p style="margin: 0; font-size: 13px; color: #ff9100;">{porc_caja:.2f}% del Portafolio Total</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c2:
            color_gan = "#00e676" if total_acciones_gan >= 0 else "#ff5252"
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Totales Acciones</p>
                    <span style="font-size: 13px; color: white;">Inv: $ {total_acciones_inv:,.2f} | Val: $ {total_mercado:,.2f}</span>
                    <h3 style="margin: 0; color: {color_gan};">$ {total_acciones_gan:,.2f} USD Ganancia</h3>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            # Color rojo si estamos por debajo del máximo de rendimiento, verde si estamos marcando récord
            color_ath = "#ff5252" if diferencia_ath < -0.01 else "#00e676"
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #00bcd4;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Máximo Histórico Ajustado</p>
                    <h3 style="margin: 0; color: white;">$ {total_ath_ajustado:,.2f} USD</h3>
                    <p style="margin: 0; font-size: 13px; color: {color_ath};">Dif: $ {diferencia_ath:,.2f} USD</p>
                </div>
            """, unsafe_allow_html=True)

        st.write("---") 

        # 5. Porcentaje sobre el total de acciones
        port['% Portafolio'] = (port['Valor Actual'] / total_acciones_val) * 100

        # --- ORDENAMIENTO POR GANANCIA ---
        port['Ganancia'] = pd.to_numeric(port['Ganancia'], errors='coerce')
        port = port.sort_values(by="Ganancia", ascending=False).reset_index(drop=True)

        # --- TABLA DE ACCIONES ---
        port_display = port.rename(columns={'ticker': 'Empresa', 'cantidad': 'Cantidad', 'monto_total': 'Cap. Invertido'})
        cols = ["Empresa", "Cantidad", "Cap. Invertido", "P. Promedio", "Cotización", "Valor Actual", "% Portafolio", "Ganancia", "% Ganancia"]
        
        def estilo_tabla(df_style):
            return df_style.format({
                'Cantidad': '{:,.8f}', 'Cap. Invertido': '$ {:,.2f}', 'P. Promedio': '$ {:,.2f}',
                'Cotización': '$ {:,.2f}', 'Valor Actual': '$ {:,.2f}', '% Portafolio': '{:.2f} %',
                'Ganancia': '$ {:,.2f}', '% Ganancia': '{:.2f} %'
            }, na_rep="-").map(
                lambda x: 'color: #00e676' if isinstance(x, (float, int)) and x > 0 else 'color: #ff5252' if isinstance(x, (float, int)) and x < 0 else '',
                subset=['Ganancia', '% Ganancia']
            )

        st.dataframe(estilo_tabla(port_display[cols].style), use_container_width=True, height="content", hide_index=True)

    else:
        st.info("Sincroniza los datos en la pestaña de Gráficos.")

    st.divider()
    
    # Formulario de Registro (Existente)
    st.subheader("Registrar Nueva Transacción")
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
    st.header("📈 Rendimiento en Moneda Local (CLP)")
    
    df_r = obtener_df(database.Resumen_cartera_diaria)
    if not df_r.empty:
        # 1. Preparación de Datos Totales
        df_total = df_r.groupby('fecha').sum(numeric_only=True).reset_index()

        # --- CÁLCULOS PARA TARJETAS EN CLP ---
        ultimo_total_clp = df_total['total_pesos'].iloc[-1]
        ultimo_retorno_clp = df_total['retorno_pesos'].iloc[-1]
        ultimo_invertido_clp = df_total['capital_invertido_pesos'].iloc[-1]
        
        # Lógica ATH Ajustado: Capital Invertido Actual + Mejor Retorno Histórico en Pesos
        max_retorno_clp_hist = df_total['retorno_pesos'].max()
        mejor_retorno_clp = max(max_retorno_clp_hist, ultimo_retorno_clp)
        
        ath_clp_ajustado = ultimo_invertido_clp + mejor_retorno_clp
        diferencia_ath_clp = ultimo_total_clp - ath_clp_ajustado
        
        # --- APARTADOS DE RESUMEN SUPERIOR (3 COLUMNAS) ---
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #ffffff;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Patrimonio Total</p>
                    <h3 style="margin: 0; color: white;">$ {ultimo_total_clp:,.0f}</h3>
                    <p style="margin: 0; font-size: 13px; color: gray;">Valor actual en CLP</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c2:
            color_ret = "#00e676" if ultimo_retorno_clp >= 0 else "#ff5252"
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Rentabilidad Total</p>
                    <h3 style="margin: 0; color: {color_ret};">$ {ultimo_retorno_clp:,.0f}</h3>
                    <p style="margin: 0; font-size: 13px; color: gray;">Ganancia/Pérdida neta</p>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            color_ath = "#00e676" if diferencia_ath_clp >= -1 else "#ff5252"
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #00bcd4;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Máximo Histórico (ATH)</p>
                    <h3 style="margin: 0; color: white;">$ {ath_clp_ajustado:,.0f}</h3>
                    <p style="margin: 0; font-size: 13px; color: {color_ath};">Dif: $ {diferencia_ath_clp:,.0f} CLP</p>
                </div>
            """, unsafe_allow_html=True)

        st.write("") # Espaciador

        # --- GRÁFICO ---
        fig_clp = go.Figure()
        
        # 2. Líneas Consolidadas
        fig_clp.add_trace(go.Scatter(x=df_total['fecha'], y=df_total['retorno_pesos'], name="TOTAL Retorno CLP", line=dict(color='#00e676', width=1.5, dash='dot'), opacity=0.4))
        fig_clp.add_trace(go.Scatter(x=df_total['fecha'], y=df_total['capital_invertido_pesos'], name="TOTAL Inv. CLP", line=dict(color='white', dash='dash', width=2)))
        fig_clp.add_trace(go.Scatter(x=df_total['fecha'], y=df_total['total_pesos'], name="TOTAL Cartera CLP", line=dict(color='white', width=2)))

        # 3. Líneas por Broker
        for br in df_r['broker'].unique():
            df_br = df_r[df_r['broker'] == br].sort_values('fecha')
            fig_clp.add_trace(go.Scatter(x=df_br['fecha'], y=df_br['retorno_pesos'], name=f"Retorno Pesos ({br})", line=dict(width=1, dash='dot')))
            fig_clp.add_trace(go.Scatter(x=df_br['fecha'], y=df_br['capital_invertido_pesos'], name=f"Inv. Pesos ({br})", line=dict(dash='dash', width=1)))
            fig_clp.add_trace(go.Scatter(x=df_br['fecha'], y=df_br['total_pesos'], name=f"Total Pesos ({br})", line=dict(width=1.5)))
        
        # 4. Configuración
        fig_clp.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="Máx")
                ]),
                bgcolor="#1E1E1E", activecolor="#2E7D32", y=1.02
            )
        )

        fig_clp.update_layout(
            template="plotly_dark", height=550, hovermode="x unified", 
            margin=dict(t=10, b=80, l=10, r=10),
            yaxis_title="Pesos ($)", yaxis=dict(tickformat=".0f"), 
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        
        fig_clp.update_traces(hovertemplate='%{y:.0f}')
        st.plotly_chart(fig_clp, use_container_width=True)
    
    st.divider()

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