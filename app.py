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
    # 1. Determinar la fecha límite según el cierre de NYSE (17:00 ET)
    tz_ny = pytz.timezone('US/Eastern')
    ahora_ny = datetime.now(tz_ny)
    fecha_limite = ahora_ny.date() if ahora_ny.hour >= 17 else ahora_ny.date() - timedelta(days=1)
    
    # Cargar datos base
    df_acc = obtener_df(database.Transacciones_acciones)
    df_div = obtener_df(database.Trasacciones_divisas)
    df_pas = obtener_df(database.Ingreso_pasivo)

    if df_acc.empty and df_div.empty and df_pas.empty:
        return st.warning("Añade transacciones, divisas o ingresos pasivos primero.")

    # Identificar todos los brokers únicos entre todas las tablas
    brokers_totales = set()
    if not df_acc.empty: brokers_totales.update(df_acc['broker'].unique())
    if not df_div.empty: brokers_totales.update(df_div['broker'].unique())
    if not df_pas.empty: brokers_totales.update(df_pas['broker'].unique())

    # --- LÓGICA DE COSTO DE POSICIÓN (WAC) ---
    cost_impacts, cash_flows = [], []
    if not df_acc.empty:
        df_acc = df_acc.sort_values(by="fecha")
        saldos_temp = {} 
        for _, row in df_acc.iterrows():
            key = (row['broker'], row['ticker'])
            if key not in saldos_temp: saldos_temp[key] = [0.0, 0.0]
            qty_act, cost_act = saldos_temp[key]
            
            if row['tipo_transaccion'] == 'compra':
                impacto = row['monto_total']
                saldos_temp[key][0] += row['cantidad']
                saldos_temp[key][1] += impacto
                cash_flows.append(-row['monto_total'])
            else: # venta
                impacto = -(cost_act * (row['cantidad'] / qty_act)) if qty_act > 0 else 0
                saldos_temp[key][0] -= row['cantidad']
                saldos_temp[key][1] += impacto
                if saldos_temp[key][0] < 1e-8: saldos_temp[key] = [0.0, 0.0]
                cash_flows.append(row['monto_total'])
            cost_impacts.append(impacto)
        df_acc['cost_impact'] = cost_impacts
        df_acc['cash_flow'] = cash_flows

    # 2. DESCARGA DE DATOS
    fechas = []
    if not df_acc.empty: fechas.append(pd.to_datetime(df_acc['fecha']).min())
    if not df_div.empty: fechas.append(pd.to_datetime(df_div['fecha']).min())
    if not df_pas.empty: fechas.append(pd.to_datetime(df_pas['fecha']).min())
    fecha_inicio = min(fechas).date()
    
    tickers = df_acc['ticker'].unique().tolist() if not df_acc.empty else []
    
    # Descarga masiva de precios y tipo de cambio
    data = yf.download(tickers, start=fecha_inicio, end=fecha_limite + timedelta(days=1))['Close'] if tickers else pd.DataFrame(index=pd.date_range(fecha_inicio, fecha_limite))
    data_usdclp = yf.download("CLP=X", start=fecha_inicio, end=fecha_limite + timedelta(days=1))['Close']
    
    # LIMPIEZA DE TABLAS DE RESUMEN
    db.query(database.Resumen_acciones_diario).delete()
    db.query(database.Resumen_acciones_diario_total).delete()
    db.query(database.Resumen_cartera_diaria).delete()
    db.query(database.Resumen_cartera_diaria_total).delete()

    for fecha_bolsa in data.index:
        f_str = fecha_bolsa.strftime("%Y-%m-%d")
        f_dt = fecha_bolsa.date()
        
        try:
            val_d = data_usdclp.loc[fecha_bolsa]
        except:
            val_d = data_usdclp.asof(fecha_bolsa)
        cambio_actual = float(val_d.iloc[0]) if hasattr(val_d, 'iloc') else float(val_d)
        
        t_hoy = df_acc[pd.to_datetime(df_acc['fecha']).dt.date <= f_dt].copy() if not df_acc.empty else pd.DataFrame()
        
        precios_dia = {}
        if not t_hoy.empty:
            t_hoy['n_cant'] = t_hoy.apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1)
            saldos = t_hoy.groupby(['broker', 'ticker']).agg({'n_cant': 'sum', 'cost_impact': 'sum'}).reset_index()
            saldos = saldos[saldos['n_cant'] > 1e-8]
            
            for tk in tickers:
                try:
                    p_val = data.loc[fecha_bolsa, tk] if isinstance(data, pd.DataFrame) else data.loc[fecha_bolsa]
                    precios_dia[tk] = float(p_val.iloc[0]) if hasattr(p_val, 'iloc') else float(p_val)
                except: precios_dia[tk] = 0.0
            
            # A y B: Resumen de Acciones (Bróker y Total)
            for _, row in saldos.iterrows():
                p = precios_dia.get(row['ticker'], 0.0)
                db.add(database.Resumen_acciones_diario(
                    fecha=f_str, broker=row['broker'], ticker=row['ticker'], precio_cierre=p,
                    monto_total=float(row['cost_impact']), cantidad=float(row['n_cant']), valor=float(p * row['n_cant'])
                ))
            
            s_tot = saldos.groupby('ticker').agg({'n_cant': 'sum', 'cost_impact': 'sum'}).reset_index()
            for _, row in s_tot.iterrows():
                p = precios_dia.get(row['ticker'], 0.0)
                db.add(database.Resumen_acciones_diario_total(
                    fecha=f_str, ticker=row['ticker'], monto_total=float(row['cost_impact']), 
                    cantidad=float(row['n_cant']), valor=float(p * row['n_cant'])
                ))
        else:
            saldos = pd.DataFrame()

        # C. Resumen Cartera por Bróker
        for br in brokers_totales:
            df_div_br = df_div[(df_div['broker'] == br) & (pd.to_datetime(df_div['fecha']).dt.date <= f_dt)] if not df_div.empty else pd.DataFrame()
            iny_usd = df_div_br.apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1).sum() if not df_div_br.empty else 0.0
            iny_clp = df_div_br.apply(lambda x: x['monto_total'] if x['tipo_transaccion'] == 'compra' else -x['monto_total'], axis=1).sum() if not df_div_br.empty else 0.0
            
            flux_usd = t_hoy[t_hoy['broker'] == br]['cash_flow'].sum() if not t_hoy.empty else 0.0
            divs_usd = df_pas[(df_pas['broker'] == br) & (pd.to_datetime(df_pas['fecha']).dt.date <= f_dt)]['monto'].sum() if not df_pas.empty else 0.0
            
            caja_usd = iny_usd + flux_usd + divs_usd
            
            s_br = saldos[saldos['broker'] == br] if not saldos.empty else pd.DataFrame()
            v_acc_usd = s_br.apply(lambda x: x['n_cant'] * precios_dia.get(x['ticker'], 0.0), axis=1).sum() if not s_br.empty else 0.0
            
            total_usd = v_acc_usd + caja_usd
            total_clp = total_usd * cambio_actual
            
            # FILTRO: Solo registrar si el total es mayor a cero
            if total_usd > 0:
                db.add(database.Resumen_cartera_diaria(
                    fecha=f_str, broker=br, valor_acciones=float(v_acc_usd), caja=float(caja_usd),
                    total=float(total_usd), total_pesos=float(total_clp),
                    capital_invertido=float(iny_usd), capital_invertido_pesos=float(iny_clp),
                    retorno=float(total_usd - iny_usd), retorno_pesos=float(total_clp - iny_clp),
                    cambio_dolar=float(cambio_actual)
                ))

        # D. Resumen Cartera Total
        t_v_acc = saldos.apply(lambda x: x['n_cant'] * precios_dia.get(x['ticker'], 0.0), axis=1).sum() if not saldos.empty else 0.0
        iny_usd_all = df_div[pd.to_datetime(df_div['fecha']).dt.date <= f_dt].apply(lambda x: x['cantidad'] if x['tipo_transaccion'] == 'compra' else -x['cantidad'], axis=1).sum() if not df_div.empty else 0.0
        iny_clp_all = df_div[pd.to_datetime(df_div['fecha']).dt.date <= f_dt].apply(lambda x: x['monto_total'] if x['tipo_transaccion'] == 'compra' else -x['monto_total'], axis=1).sum() if not df_div.empty else 0.0
        caja_all = iny_usd_all + (t_hoy['cash_flow'].sum() if not t_hoy.empty else 0.0) + (df_pas[pd.to_datetime(df_pas['fecha']).dt.date <= f_dt]['monto'].sum() if not df_pas.empty else 0.0)
        
        total_usd_all = t_v_acc + caja_all
        total_clp_all = total_usd_all * cambio_actual
        
        # FILTRO: Solo registrar si el total global es mayor a cero
        if total_usd_all > 0:
            db.add(database.Resumen_cartera_diaria_total(
                fecha=f_str, valor_acciones=float(t_v_acc), caja=float(caja_all),
                total=float(total_usd_all), total_pesos=float(total_clp_all),
                capital_invertido=float(iny_usd_all), capital_invertido_pesos=float(iny_clp_all),
                retorno=float(total_usd_all - iny_usd_all), retorno_pesos=float(total_clp_all - iny_clp_all),
                cambio_dolar=float(cambio_actual)
            ))
        
    db.commit()
    st.success(f"Sincronización completada. Datos procesados hasta: {fecha_limite}")

# --- INTERFAZ ---
st.title("📂 Mi Portafolio de Inversiones")
tab_acciones, tab_graficos, tab_divisas, tab_pasivos = st.tabs([
    "📊 Portafolio", "📈 Histórico Dólares", "💵 Histórico Pesos", "💰 Ingresos Pasivos"
])

# 1. PORTAFOLIO
with tab_acciones:
    st.header("📊 Portafolio")
    
    if st.button("🔄 Actualizar Precios y Datos"):
        st.rerun()

    df_h = obtener_df(database.Resumen_acciones_diario)
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
        
        # 4. Cálculos Globales
        total_efectivo = df_r_hoy['caja'].sum()
        total_acciones_val = port['Valor Actual'].sum()
        total_mercado = total_acciones_val + total_efectivo
        total_deposito = df_r_hoy['capital_invertido'].sum()
        ganancia_total_cartera = total_mercado - total_deposito
        
        # ATH basado en Retorno
        max_retorno_hist = df_r['retorno'].max()
        retorno_actual = total_mercado - total_deposito
        mejor_retorno = max(max_retorno_hist, retorno_actual)
        diferencia_ath_retorno = retorno_actual - mejor_retorno

        # --- SECCIÓN: CAJA POR BRÓKER ---
        st.write("### 🏦 Caja por Bróker")
        cols_caja = st.columns(len(df_r_hoy))
        for i, (_, row) in enumerate(df_r_hoy.iterrows()):
            with cols_caja[i]:
                st.metric(label=f"Caja {row['broker']}", value=f"$ {row['caja']:,.2f}")

        st.write("") 

        # --- SECCIÓN: TARJETAS DE RESUMEN GLOBAL ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9100;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Efectivo Total (Caja)</p>
                    <h3 style="margin: 0; color: white;">$ {total_efectivo:,.2f} USD</h3>
                    <p style="margin: 0; font-size: 13px; color: #ff9100;">{(total_efectivo / total_mercado * 100):.2f}% del Portafolio</p>
                </div>
            """, unsafe_allow_html=True)
            
        with c2:
            color_gan = "#00e676" if ganancia_total_cartera >= 0 else "#ff5252"
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Ganancia Total Portafolio</p>
                    <span style="font-size: 13px; color: white;">Inv: $ {total_deposito:,.2f} | Val: $ {total_mercado:,.2f}</span>
                    <h3 style="margin: 0; color: {color_gan};">$ {ganancia_total_cartera:,.2f} USD</h3>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            color_ath = "#00e676" if diferencia_ath_retorno >= -0.01 else "#ff5252"
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #00bcd4;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Retorno Máximo (ATH)</p>
                    <h3 style="margin: 0; color: white;">$ {mejor_retorno:,.2f} USD</h3>
                    <p style="margin: 0; font-size: 13px; color: {color_ath};">Dif. vs ATH: $ {diferencia_ath_retorno:,.2f} USD</p>
                </div>
            """, unsafe_allow_html=True)

        st.write("---") 

        # 5. Tabla de Acciones
        port['% Portafolio'] = (port['Valor Actual'] / total_mercado) * 100
        port = port.sort_values(by="Ganancia", ascending=False).reset_index(drop=True)

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

        # --- SECCIÓN: ANÁLISIS VISUAL (GRÁFICOS VERTICALES) ---
        st.write("### 📉 Análisis Visual")

        # Gráfico 1: Barras Comparativas (Inversión vs Valor Actual + Caja)
        labels_bar = port['ticker'].tolist() + [f"Caja {b}" for b in df_r_hoy['broker'].tolist()]
        val_inv_bar = port['monto_total'].tolist() + df_r_hoy['caja'].tolist()
        val_act_bar = port['Valor Actual'].tolist() + df_r_hoy['caja'].tolist()
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Cap. Invertido', x=labels_bar, y=val_inv_bar, marker_color='#607D8B'),
            go.Bar(name='Valor Actual', x=labels_bar, y=val_act_bar, marker_color='#00e676')
        ])
        fig_bar.update_layout(
            barmode='group', template="plotly_dark", title="Comparativa Inversión vs Valor Actual (USD)",
            margin=dict(t=50, b=20, l=20, r=20), height=500, 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # 2. Gráfico Treemap de Distribución Patrimonial Corregido
        # Definimos la jerarquía para que los hijos sumen exactamente el total del padre
        labels_tree = ["Total Portafolio", "Acciones", "Efectivo"]
        parents_tree = ["", "Total Portafolio", "Total Portafolio"]
        values_tree = [total_mercado, total_acciones_val, total_efectivo]

        # Agregar Acciones a la jerarquía
        for _, row in port.iterrows():
            if row['Valor Actual'] > 0:
                labels_tree.append(row['ticker'])
                parents_tree.append("Acciones")
                values_tree.append(row['Valor Actual'])

        # Agregar Cajas a la jerarquía
        for _, row in df_r_hoy.iterrows():
            if row['caja'] > 0:
                labels_tree.append(f"Caja {row['broker']}")
                parents_tree.append("Efectivo")
                values_tree.append(row['caja'])
        
        fig_tree = go.Figure(go.Treemap(
            labels=labels_tree,
            parents=parents_tree,
            values=values_tree,
            branchvalues="total", # Crucial: hace que las cajas llenen todo el espacio
            textinfo="label+value+percent root", # % calculado sobre el total del portafolio
            marker_colorscale='Greens',
            hovertemplate='<b>%{label}</b><br>Valor: $ %{value:,.2f}<br>% del Portafolio: %{percentRoot:.2%}'
        ))
        fig_tree.update_layout(
            template="plotly_dark", title="Distribución de Activos (%)",
            margin=dict(t=50, b=20, l=20, r=20), height=600
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    else:
        st.info("Sincroniza los datos en la pestaña de Gráficos.")

    st.divider()
    
    # Formulario y Tabla de Transacciones
    st.subheader("Registrar Nueva Transacción")
    with st.form("form_acc"):
        col1, col2, col3 = st.columns(3)
        f = col1.date_input("Fecha"); t = col1.text_input("Ticker").upper()
        tipo = col2.selectbox("Operación", ["compra", "venta"]); br = col2.selectbox("Broker", ["Racional", "Zesty", "Fintual"])
        mt = col3.number_input("Monto Total (USD)"); pr = col3.number_input("Precio Acción"); ct = col3.number_input("Cantidad", format="%.8f")
        if st.form_submit_button("Guardar"):
            db.add(database.Transacciones_acciones(fecha=str(f), broker=br, tipo_transaccion=tipo, ticker=t, monto_total=mt, precio=pr, cantidad=ct))
            db.commit(); st.rerun()
            
    df_trans = obtener_df(database.Transacciones_acciones)
    if not df_trans.empty:
        df_trans = df_trans.sort_values(by="fecha", ascending=False)
    mostrar_tabla_estilizada(df_trans, "acciones")

# 2. HISTORICO DE ACCIONES Y TOTALES EN DOLARES
with tab_graficos:
    st.header("Histórico de Acciones y Totales")
    if st.button("🔄 Sincronizar Portafolio"):
        with st.spinner("Procesando datos..."): sincronizar_todo(db)
        st.rerun()

    # Cargamos tanto las tablas por bróker como las de totales
    df_r = obtener_df(database.Resumen_cartera_diaria)
    df_h = obtener_df(database.Resumen_acciones_diario)
    df_r_total = obtener_df(database.Resumen_cartera_diaria_total)
    df_h_total = obtener_df(database.Resumen_acciones_diario_total)

    if not df_r.empty:
        # --- FILTROS ---
        brokers_disponibles = df_r['broker'].unique()
        c1, c2, c3, c4 = st.columns([1, 1, 2, 0.5])
        br_sel = c1.multiselect("Bróker", brokers_disponibles, default=list(brokers_disponibles))
        met_sel = c2.multiselect("Métricas Cartera", ["Total", "Retorno", "Caja", "Capital Invertido"], default=["Retorno"])
        tk_sel = c3.multiselect("Filtrar Acción (Selecciona para aislar)", df_h['ticker'].unique())
        with c4:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            ver_acciones = st.checkbox("Ver Acciones", value=True)

        # --- OPTIMIZACIÓN: Selección de Tabla de Datos ---
        # Si selecciona todos los brókeres, usamos la tabla de totales para mayor velocidad
        if len(br_sel) == len(brokers_disponibles):
            df_p = df_r_total.copy()
        else:
            df_p = df_r[df_r['broker'].isin(br_sel)].groupby('fecha').sum(numeric_only=True).reset_index()

        # --- RESUMEN DE VALORES CENTRADOS ---
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
        
        ultima_f = pd.to_datetime(df_p['fecha']).max()
        inicio_f = ultima_f - pd.DateOffset(years=1)
        primera_f = pd.to_datetime(df_p['fecha']).min()
        if inicio_f < primera_f: inicio_f = primera_f

        # Métricas principales
        if "Total" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['total'], name="TOTAL", line=dict(color='white', width=1.5)))
        if "Retorno" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['retorno'], name="RETORNO", line=dict(color='#00e676', width=1.5)))
        if "Caja" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['caja'], name="CAJA", line=dict(dash='dash', color='#ff9100', width=1.5)))
        if "Capital Invertido" in met_sel: fig.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['capital_invertido'], name="CAP. INVERTIDO", line=dict(color='gray', dash='dot', width=1.5)))

        # Lógica de Acciones (Usa tablas de totales si aplica)
        if ver_acciones:
            tickers_a_mostrar = tk_sel if tk_sel else df_h['ticker'].unique()
            for tk in tickers_a_mostrar:
                if len(br_sel) == len(brokers_disponibles):
                    # Usamos Resumen_acciones_diario_total para filtrar por ticker global
                    df_tk = df_h_total[df_h_total['ticker'] == tk].copy()
                else:
                    df_tk = df_h[(df_h['ticker'] == tk) & (df_h['broker'].isin(br_sel))].groupby('fecha').sum(numeric_only=True).reset_index()
                
                df_tk = pd.merge(df_p[['fecha']], df_tk, on='fecha', how='left')
                
                fig.add_trace(go.Scatter(x=df_tk['fecha'], y=df_tk['valor'], name=f"Valor: {tk}", line=dict(width=1), connectgaps=False))

                if tk_sel:
                    fig.add_trace(go.Scatter(x=df_tk['fecha'], y=df_tk['monto_total'], name=f"Inv: {tk}", line=dict(width=1, dash='dot'), opacity=0.6, connectgaps=False))

        # Estilo del Gráfico
        fig.update_xaxes(
            range=[inicio_f, ultima_f],
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="Máx")
                ]), bgcolor="#1E1E1E", activecolor="#2E7D32", y=1.01
            )
        )

        fig.update_layout(
            template="plotly_dark", height=550, hovermode="x unified",
            margin=dict(t=10, b=80, l=10, r=10), yaxis=dict(tickformat=".2f"),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        fig.update_traces(hovertemplate='%{y:.2f}')
        st.plotly_chart(fig, use_container_width=True)

        # --- TABLA DE DETALLE DIARIO (Solo Totales de Acciones) ---
        st.subheader("📜 Detalle Diario de Acciones")
        if not df_h_total.empty:
            # Mostramos la tabla de totales ordenada por fecha descendente
            df_display = df_h_total.sort_values(by="fecha", ascending=False)
            mostrar_tabla_estilizada(df_display, "historico")

# 3. HISTORICO DE TOTALES EN PESOS
with tab_divisas:
    st.header("📈 Histórico en Pesos")
    
    if st.button("🔄 Sincronizar Portafolio", key="sync_divisas_final"):
        with st.spinner("Actualizando datos..."):
            sincronizar_todo(db)
        st.rerun()

    df_r = obtener_df(database.Resumen_cartera_diaria)
    df_total = obtener_df(database.Resumen_cartera_diaria_total)

    if not df_total.empty:
        # --- CÁLCULOS PARA TARJETAS EN CLP ---
        ultimo_resumen = df_total.iloc[-1]
        ultimo_total_clp = ultimo_resumen['total_pesos']
        ultimo_retorno_clp = ultimo_resumen['retorno_pesos']
        ultimo_invertido_clp = ultimo_resumen['capital_invertido_pesos']
        
        # ATH basado en el Rendimiento Máximo
        max_retorno_clp_hist = df_total['retorno_pesos'].max()
        mejor_retorno_clp = max(max_retorno_clp_hist, ultimo_retorno_clp)
        diferencia_rendimiento = ultimo_retorno_clp - mejor_retorno_clp
        
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
            color_ath = "#00e676" if diferencia_rendimiento >= -1 else "#ff5252"
            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #00bcd4;">
                    <p style="margin: 0; font-size: 14px; color: gray;">Rendimiento Máximo (ATH)</p>
                    <h3 style="margin: 0; color: white;">$ {mejor_retorno_clp:,.0f}</h3>
                    <p style="margin: 0; font-size: 13px; color: {color_ath};">Dif vs ATH: $ {diferencia_rendimiento:,.0f} CLP</p>
                </div>
            """, unsafe_allow_html=True)

        st.write("") 

        # --- GRÁFICO MULTI-LÍNEA (Sin puntos) ---
        fig_clp = go.Figure()
        
        # 1. LÍNEAS TOTALES
        fig_clp.add_trace(go.Scatter(
            x=df_total['fecha'], y=df_total['total_pesos'], 
            name="TOTAL Cartera", mode='lines', line=dict(color='white', width=3)
        ))
        fig_clp.add_trace(go.Scatter(
            x=df_total['fecha'], y=df_total['capital_invertido_pesos'], 
            name="TOTAL Invertido", mode='lines', line=dict(color='gray', width=2, dash='dash')
        ))
        fig_clp.add_trace(go.Scatter(
            x=df_total['fecha'], y=df_total['retorno_pesos'], 
            name="TOTAL Retorno", mode='lines', line=dict(color='#00e676', width=2, dash='dot'), opacity=0.7
        ))

        # 2. LÍNEAS POR BRÓKER
        for br in df_r['broker'].unique():
            df_br = df_r[df_r['broker'] == br].sort_values('fecha')
            fig_clp.add_trace(go.Scatter(
                x=df_br['fecha'], y=df_br['total_pesos'], 
                name=f"Total ({br})", mode='lines', line=dict(width=1.5), opacity=0.8
            ))
            fig_clp.add_trace(go.Scatter(
                x=df_br['fecha'], y=df_br['capital_invertido_pesos'], 
                name=f"Inv ({br})", mode='lines', line=dict(width=1, dash='dash'), opacity=0.5
            ))
            fig_clp.add_trace(go.Scatter(
                x=df_br['fecha'], y=df_br['retorno_pesos'], 
                name=f"Retorno ({br})", mode='lines', line=dict(width=1, dash='dot'), opacity=0.5
            ))

        # 3. Lógica de Rango Temporal Inicial (1 Año)
        ultima_f = pd.to_datetime(df_total['fecha']).max()
        inicio_f = ultima_f - pd.DateOffset(years=1)
        primera_f = pd.to_datetime(df_total['fecha']).min()
        if inicio_f < primera_f:
            inicio_f = primera_f
        
        # 4. Configuración de Botones y Selector de Rango
        fig_clp.update_xaxes(
            range=[inicio_f, ultima_f], # Vista inicial de 1 año
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
                y=1.02
            )
        )

        fig_clp.update_layout(
            template="plotly_dark", height=650, hovermode="x unified", 
            margin=dict(t=10, b=80, l=10, r=10),
            yaxis_title="Pesos ($)", yaxis=dict(tickformat=", .0f"), 
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        
        fig_clp.update_traces(hovertemplate='%{y:,.0f}')
        st.plotly_chart(fig_clp, use_container_width=True)
    
    st.divider()

    # --- HISTORIAL Y REGISTRO ---
    st.header("Historial de Divisas")
    with st.form("f_div_new"):
        c1, c2, c3 = st.columns(3)
        fd = c1.date_input("Fecha"); bd = c1.selectbox("Broker", ["Racional", "Zesty", "Fintual", "BancoEstado"])
        td = c2.selectbox("Tipo", ["compra", "venta"]); mc = c2.number_input("Monto CLP")
        tc = c3.number_input("Tipo Cambio"); cu = c3.number_input("Cantidad USD")
        if st.form_submit_button("Registrar"):
            db.add(database.Trasacciones_divisas(fecha=str(fd), broker=bd, tipo_transaccion=td, monto_total=mc, precio=tc, cantidad=cu))
            db.commit(); st.rerun()
            
    df_div_hist = obtener_df(database.Trasacciones_divisas)
    if not df_div_hist.empty:
        mostrar_tabla_estilizada(df_div_hist.sort_values(by="fecha", ascending=False), "divisas")

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
    
    # Se agrega el ordenamiento por fecha de forma descendente
    df_pasivos = obtener_df(database.Ingreso_pasivo)
    if not df_pasivos.empty:
        df_pasivos = df_pasivos.sort_values(by="fecha", ascending=False)
    
    mostrar_tabla_estilizada(df_pasivos, "pasivos")

# 5. CALCULOS