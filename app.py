import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="VoglioVolare BI", page_icon="✈️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        padding: 18px; border-radius: 12px; border-left: 4px solid #00d4aa;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #00d4aa; }
    .metric-label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; }
    .highlight-box {
        background: linear-gradient(135deg, #0d3320 0%, #0d2137 100%);
        border: 1px solid #00d4aa; border-radius: 10px; padding: 15px; margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #3d1f1f 0%, #0d2137 100%);
        border: 1px solid #ff6b6b; border-radius: 10px; padding: 15px; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

COLORS = {
    'primary': '#00d4aa', 'secondary': '#1e3a5f', 'accent': '#e94560',
    'warning': '#ff6b6b', 'text': '#ccd6f6', 'muted': '#8892b0',
    'year_colors': ['#00d4aa', '#e94560', '#f39c12', '#3498db', '#9b59b6']
}

# --- MAPPATURA MONDAY ---
COLUMN_MAPPING = {
    "cliente": "dup__of_note", "paese": "dup__of_cliente", "assegnato": "stato",
    "status": "status7", "arrivo_richiesta": "dup__of_last_update",
    "invio_offerta": "data", "date_viaggio": "timeline7", "n_pax": "numeri2",
    "fatturato_tot": "numbers4", "costo_tot": "numbers1", "margine_lordo": "formula",
}

# --- FUNZIONI API ---
def fetch_monday_data(api_key, board_id):
    url = "https://api.monday.com/v2"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    all_items, cursor = [], None
    
    while True:
        q = f'query {{ boards (ids: {board_id}) {{ items_page (limit: 500{f", cursor: \"{cursor}\"" if cursor else ""}) {{ cursor items {{ name group {{ title }} column_values {{ id text value }} }} }} }} }}'
        try:
            r = requests.post(url, json={'query': q}, headers=headers)
            if r.status_code == 200:
                data = r.json()['data']['boards'][0]['items_page']
                all_items.extend(data['items'])
                cursor = data.get('cursor')
                if not cursor: break
            else: return None
        except: return None
    return {'data': {'boards': [{'items_page': {'items': all_items}}]}}

def parse_currency(value):
    if value is None or value == "": return 0.0
    if isinstance(value, (int, float)): return float(value)
    s = str(value).replace("€", "").replace("$", "").replace("£", "").strip()
    if not s: return 0.0
    try:
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".") if s.find(".") < s.find(",") else s
        elif "," in s: s = s.replace(",", ".")
        return float(s)
    except: return 0.0

def process_monday_data(raw_data):
    try: items = raw_data['data']['boards'][0]['items_page']['items']
    except: return pd.DataFrame()
    
    rows = []
    for item in items:
        row = {"Nome": item['name']}
        try: row["GRUPPO"] = item['group']['title']
        except: row["GRUPPO"] = "N/A"
        
        col_text = {c['id']: c['text'] for c in item['column_values']}
        col_val = {}
        for c in item['column_values']:
            if c.get('value'):
                try:
                    v = json.loads(c['value'])
                    col_val[c['id']] = v.get('value', v) if isinstance(v, dict) else v
                except: pass
        
        row["CLIENTE"] = col_text.get(COLUMN_MAPPING["cliente"], "")
        row["PAESE"] = col_text.get(COLUMN_MAPPING["paese"], "")
        row["ASSEGNATO"] = col_text.get(COLUMN_MAPPING["assegnato"], "")
        row["STATUS"] = col_text.get(COLUMN_MAPPING["status"], "")
        row["ARRIVO RICHIESTA"] = col_text.get(COLUMN_MAPPING["arrivo_richiesta"], "")
        row["INVIO OFFERTA"] = col_text.get(COLUMN_MAPPING["invio_offerta"], "")
        
        viaggio = col_text.get(COLUMN_MAPPING["date_viaggio"], "")
        row["DATA VIAGGIO"] = viaggio.split(" - ")[0] if " - " in viaggio else viaggio
        
        for k, m in [("N DI PAX", "n_pax"), ("FATTURATO TOT", "fatturato_tot"), ("COSTO TOT", "costo_tot"), ("MARGINE LORDO", "margine_lordo")]:
            row[k] = parse_currency(col_val.get(COLUMN_MAPPING[m]) or col_text.get(COLUMN_MAPPING[m], 0))
        
        if row["MARGINE LORDO"] == 0 and row["FATTURATO TOT"] > 0 and row["COSTO TOT"] > 0:
            row["MARGINE LORDO"] = row["FATTURATO TOT"] - row["COSTO TOT"]
        row["MARGINE %"] = (row["MARGINE LORDO"] / row["FATTURATO TOT"] * 100) if row["FATTURATO TOT"] > 0 else 0
        rows.append(row)
    
    df = pd.DataFrame(rows)
    for c in ["ARRIVO RICHIESTA", "INVIO OFFERTA", "DATA VIAGGIO"]:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    
    # Date richiesta
    df["ANNO_RICHIESTA"] = df["ARRIVO RICHIESTA"].dt.year
    df["MESE_RICHIESTA"] = df["ARRIVO RICHIESTA"].dt.month
    df["SETTIMANA_RICHIESTA"] = df["ARRIVO RICHIESTA"].dt.isocalendar().week
    
    # Date viaggio
    df["ANNO_VIAGGIO"] = df["DATA VIAGGIO"].dt.year
    df["MESE_VIAGGIO"] = df["DATA VIAGGIO"].dt.month
    
    df["GIORNI_QUOTAZIONE"] = (df["INVIO OFFERTA"] - df["ARRIVO RICHIESTA"]).dt.days
    
    # Tipo pratica dal gruppo
    # REALIZZATI e CONFERMATI = CONVERTITA (pratica vinta)
    # SOSPESI e ANNULLATI = NON CONVERTITA (pratica persa)
    # IN LAVORAZIONE = ancora aperta
    def tipo(g):
        g = str(g).upper()
        if "REALIZZAT" in g: return "CONVERTITA"
        if "CONFERMAT" in g: return "CONVERTITA"
        if "SOSPES" in g: return "NON CONVERTITA"
        if "ANNULLAT" in g: return "NON CONVERTITA"
        if "LAVORAZIONE" in g: return "IN LAVORAZIONE"
        return "ALTRO"
    df["TIPO"] = df["GRUPPO"].apply(tipo)
    
    return df

def fmt_curr(v):
    if v >= 1e6: return f"€{v/1e6:.2f}M"
    if v >= 1e3: return f"€{v/1e3:.0f}K"
    return f"€{v:.0f}"

# --- INTERFACCIA ---
st.markdown("# ✈️ VoglioVolare - Dashboard Operativa")

with st.sidebar:
    st.markdown("### ⚙️ Connessione")
    api_key = st.text_input("API Key Monday", type="password")
    board_id = st.text_input("Board ID")
    if st.button("🚀 Carica", type="primary", use_container_width=True):
        if api_key and board_id:
            with st.spinner("Caricamento..."):
                raw = fetch_monday_data(api_key, board_id)
                if raw:
                    st.session_state["df"] = process_monday_data(raw)
                    st.success(f"✅ {len(st.session_state['df'])} pratiche")
                    st.rerun()

if "df" not in st.session_state or st.session_state["df"] is None:
    st.info("👈 Connettiti a Monday per iniziare")
    st.stop()

df = st.session_state["df"]

# --- ANNI DISPONIBILI ---
anni_viaggio = sorted([int(a) for a in df["ANNO_VIAGGIO"].dropna().unique() if a >= 2020])
anni_richiesta = sorted([int(a) for a in df["ANNO_RICHIESTA"].dropna().unique() if a >= 2020])
anno_corrente = datetime.now().year

with st.sidebar:
    st.markdown("---")
    st.markdown(f"**{len(df)} pratiche totali**")

# --- KPI HEADER GLOBALI ---
st.markdown("---")

# Calcoli globali
df_convertite = df[df["TIPO"] == "CONVERTITA"]
df_non_conv = df[df["TIPO"] == "NON CONVERTITA"]
df_lavorazione = df[df["TIPO"] == "IN LAVORAZIONE"]
df_altro = df[df["TIPO"] == "ALTRO"]
# CR = Convertite / (Totale - In Lavorazione) = Convertite / pratiche chiuse
pratiche_chiuse = len(df) - len(df_lavorazione)
cr_globale = (len(df_convertite) / pratiche_chiuse * 100) if pratiche_chiuse > 0 else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("📥 Pratiche Totali", len(df))
c2.metric("✅ Convertite", len(df_convertite))
c3.metric("❌ Sospese/Annullate", len(df_non_conv) + len(df_altro))
c4.metric("🔄 In Lavorazione", len(df_lavorazione))
c5.metric("🎯 CR Globale", f"{cr_globale:.1f}%")
c6.metric("💰 Fatturato Tot.", fmt_curr(df_convertite["FATTURATO TOT"].sum()))

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📅 Arrivo Pratiche",
    "✈️ Viaggi per Anno", 
    "📊 Benchmark",
    "🔮 Outlook 2026",
    "🌍 Mercati",
    "👥 Team"
])

# =============================================================================
# TAB 1: ARRIVO PRATICHE (per data richiesta)
# =============================================================================
with tab1:
    st.markdown("### 📅 Analisi Arrivo Richieste")
    st.caption("Quando arrivano le richieste dei clienti")
    
    anno_sel = st.selectbox("Anno arrivo richieste", anni_richiesta, index=len(anni_richiesta)-1 if anni_richiesta else 0)
    df_anno_req = df[df["ANNO_RICHIESTA"] == anno_sel]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##### Pratiche Arrivate per Mese")
        monthly = df_anno_req.groupby("MESE_RICHIESTA").agg({"Nome": "count", "FATTURATO TOT": "sum"}).reset_index()
        monthly.columns = ["Mese", "Pratiche", "Valore Potenziale"]
        mesi = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
        monthly["Mese_Nome"] = monthly["Mese"].apply(lambda x: mesi[int(x)-1] if pd.notna(x) and 1 <= x <= 12 else "")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly["Mese_Nome"], y=monthly["Pratiche"], name="N° Pratiche", marker_color=COLORS['primary']), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["Mese_Nome"], y=monthly["Valore Potenziale"], name="Valore Potenziale", mode="lines+markers", line=dict(color=COLORS['accent'], width=2)), secondary_y=True)
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Riepilogo")
        conv = df_anno_req[df_anno_req["TIPO"] == "CONVERTITA"]
        sosp_ann = df_anno_req[df_anno_req["TIPO"] == "NON CONVERTITA"]
        lav = df_anno_req[df_anno_req["TIPO"] == "IN LAVORAZIONE"]
        altro = df_anno_req[df_anno_req["TIPO"] == "ALTRO"]
        
        # CR = Convertite / (Totale - In Lavorazione)
        pratiche_chiuse = len(df_anno_req) - len(lav)
        cr = (len(conv) / pratiche_chiuse * 100) if pratiche_chiuse > 0 else 0
        
        st.metric("📥 Richieste Totali", len(df_anno_req))
        st.markdown("---")
        st.metric("✅ Convertite", len(conv))
        st.metric("❌ Sospese/Annullate/Altro", len(sosp_ann) + len(altro))
        st.metric("🔄 In Lavorazione", len(lav))
        st.markdown("---")
        st.metric("🎯 Conversion Rate", f"{cr:.1f}%", help="Convertite / (Totale - In Lavorazione)")
        st.metric("💰 Fatturato Generato", fmt_curr(conv["FATTURATO TOT"].sum()))
    
    # Heatmap
    st.markdown("##### Heatmap Arrivo Pratiche")
    heatmap_data = df_anno_req.groupby(["MESE_RICHIESTA", "SETTIMANA_RICHIESTA"]).size().reset_index(name="Pratiche")
    if len(heatmap_data) > 0:
        pivot = heatmap_data.pivot(index="SETTIMANA_RICHIESTA", columns="MESE_RICHIESTA", values="Pratiche").fillna(0)
        pivot.columns = [mesi[int(c)-1] if 1 <= c <= 12 else str(c) for c in pivot.columns]
        fig_heat = px.imshow(pivot, color_continuous_scale=[[0, '#0d2137'], [0.5, '#1e3a5f'], [1, '#00d4aa']], aspect="auto")
        fig_heat.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']))
        st.plotly_chart(fig_heat, use_container_width=True)

# =============================================================================
# TAB 2: VIAGGI PER ANNO (per data viaggio)
# =============================================================================
with tab2:
    st.markdown("### ✈️ Pratiche per Anno di Viaggio")
    st.caption("Quando i clienti viaggiano effettivamente")
    
    anno_viaggio_sel = st.selectbox("Anno viaggio", anni_viaggio, index=len(anni_viaggio)-1 if anni_viaggio else 0)
    df_viaggio = df[df["ANNO_VIAGGIO"] == anno_viaggio_sel]
    
    # Suddivisione per stato
    conv_v = df_viaggio[df_viaggio["TIPO"] == "CONVERTITA"]
    non_conv_v = df_viaggio[df_viaggio["TIPO"] == "NON CONVERTITA"]
    lav_v = df_viaggio[df_viaggio["TIPO"] == "IN LAVORAZIONE"]
    altro_v = df_viaggio[df_viaggio["TIPO"] == "ALTRO"]
    # CR = Convertite / (Totale - In Lavorazione)
    chiuse_v = len(df_viaggio) - len(lav_v)
    cr_v = (len(conv_v) / chiuse_v * 100) if chiuse_v > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"✅ Convertite {anno_viaggio_sel}", len(conv_v), fmt_curr(conv_v["FATTURATO TOT"].sum()))
    col2.metric(f"❌ Sospese/Ann./Altro {anno_viaggio_sel}", len(non_conv_v) + len(altro_v), fmt_curr(non_conv_v["FATTURATO TOT"].sum()))
    col3.metric(f"🔄 In Lavorazione {anno_viaggio_sel}", len(lav_v), fmt_curr(lav_v["FATTURATO TOT"].sum()))
    col4.metric(f"🎯 CR {anno_viaggio_sel}", f"{cr_v:.1f}%")
    
    # Grafico mensile viaggi
    st.markdown("##### Distribuzione Viaggi per Mese")
    
    monthly_viaggio = df_viaggio.groupby(["MESE_VIAGGIO", "TIPO"]).agg({"Nome": "count", "FATTURATO TOT": "sum"}).reset_index()
    monthly_viaggio.columns = ["Mese", "Tipo", "Pratiche", "Fatturato"]
    mesi = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
    monthly_viaggio["Mese_Nome"] = monthly_viaggio["Mese"].apply(lambda x: mesi[int(x)-1] if pd.notna(x) and 1 <= x <= 12 else "")
    
    # Rinomina tipi per display
    tipo_labels = {"CONVERTITA": "Convertite", "NON CONVERTITA": "Sospese/Ann.", "IN LAVORAZIONE": "In Lavorazione", "ALTRO": "Altro"}
    monthly_viaggio["Tipo"] = monthly_viaggio["Tipo"].map(tipo_labels).fillna(monthly_viaggio["Tipo"])
    
    color_map = {"Convertite": COLORS['primary'], "Sospese/Ann.": COLORS['warning'], "In Lavorazione": COLORS['accent'], "Altro": COLORS['muted']}
    
    fig = px.bar(monthly_viaggio, x="Mese_Nome", y="Pratiche", color="Tipo", barmode="stack", 
                 color_discrete_map=color_map, category_orders={"Mese_Nome": mesi})
    fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabella dettaglio
    st.markdown("##### Dettaglio Pratiche")
    df_viaggio_display = df_viaggio[["Nome", "CLIENTE", "PAESE", "TIPO", "DATA VIAGGIO", "FATTURATO TOT", "N DI PAX"]].copy()
    tipo_labels = {"CONVERTITA": "Convertita", "NON CONVERTITA": "Sospesa/Ann.", "IN LAVORAZIONE": "In Lavorazione", "ALTRO": "Altro"}
    df_viaggio_display["TIPO"] = df_viaggio_display["TIPO"].map(tipo_labels).fillna(df_viaggio_display["TIPO"])
    df_viaggio_display["FATTURATO TOT"] = df_viaggio_display["FATTURATO TOT"].apply(lambda x: f"€{x:,.0f}")
    df_viaggio_display["DATA VIAGGIO"] = df_viaggio_display["DATA VIAGGIO"].dt.strftime("%d/%m/%Y").fillna("-")
    st.dataframe(df_viaggio_display.sort_values("TIPO"), use_container_width=True, hide_index=True, height=300)

# =============================================================================
# TAB 3: BENCHMARK
# =============================================================================
with tab3:
    st.markdown("### 📊 Benchmark tra Anni")
    
    # Calcola metriche per anno viaggio
    benchmark_data = []
    for anno in anni_viaggio:
        df_a = df[df["ANNO_VIAGGIO"] == anno]
        df_rich_a = df[df["ANNO_RICHIESTA"] == anno]
        conv = df_a[df_a["TIPO"] == "CONVERTITA"]
        lav = df_a[df_a["TIPO"] == "IN LAVORAZIONE"]
        # Pratiche chiuse = Totale - In Lavorazione
        chiuse = len(df_a) - len(lav)
        non_conv_count = chiuse - len(conv)
        
        benchmark_data.append({
            "Anno": anno,
            "Richieste Arrivate": len(df_rich_a),
            "Viaggi Totali": len(df_a),
            "Convertite": len(conv),
            "Non Convertite": non_conv_count,
            "Conversion Rate %": (len(conv) / chiuse * 100) if chiuse > 0 else 0,
            "Fatturato": conv["FATTURATO TOT"].sum(),
            "Margine": conv["MARGINE LORDO"].sum(),
            "Margine %": (conv["MARGINE LORDO"].sum() / conv["FATTURATO TOT"].sum() * 100) if conv["FATTURATO TOT"].sum() > 0 else 0,
            "PAX": conv["N DI PAX"].sum(),
            "Fatt. Medio": conv["FATTURATO TOT"].sum() / len(conv) if len(conv) > 0 else 0
        })
    
    df_bench = pd.DataFrame(benchmark_data)
    
    col_sel, col_chart = st.columns([1, 3])
    
    with col_sel:
        metrica = st.radio("Metrica", [
            "Fatturato", "Richieste Arrivate", "Convertite", 
            "Conversion Rate %", "Margine", "Margine %", "PAX", "Fatt. Medio"
        ])
    
    with col_chart:
        fig = go.Figure()
        if metrica in ["Fatturato", "Margine", "Fatt. Medio"]:
            fig.add_trace(go.Bar(x=df_bench["Anno"].astype(str), y=df_bench[metrica], marker_color=COLORS['primary'], text=df_bench[metrica].apply(fmt_curr), textposition='outside'))
        elif metrica in ["Conversion Rate %", "Margine %"]:
            fig.add_trace(go.Bar(x=df_bench["Anno"].astype(str), y=df_bench[metrica], marker_color=COLORS['primary'], text=df_bench[metrica].round(1).astype(str) + "%", textposition='outside'))
        else:
            fig.add_trace(go.Bar(x=df_bench["Anno"].astype(str), y=df_bench[metrica], marker_color=COLORS['primary'], text=df_bench[metrica].astype(int), textposition='outside'))
        fig.update_layout(title=f"{metrica} per Anno", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
        st.plotly_chart(fig, use_container_width=True)
    
    # Variazioni YoY
    st.markdown("##### Variazioni Anno su Anno")
    if len(df_bench) >= 2:
        df_yoy = df_bench.copy()
        for col in ["Fatturato", "Richieste Arrivate", "Convertite", "Conversion Rate %", "PAX"]:
            df_yoy[f"{col} Δ%"] = df_yoy[col].pct_change() * 100
        
        cols_show = ["Anno", "Fatturato", "Fatturato Δ%", "Convertite", "Convertite Δ%", "Conversion Rate %", "Conversion Rate % Δ%"]
        df_yoy_disp = df_yoy[[c for c in cols_show if c in df_yoy.columns]].copy()
        df_yoy_disp["Fatturato"] = df_yoy_disp["Fatturato"].apply(fmt_curr)
        for col in df_yoy_disp.columns:
            if "Δ%" in col:
                df_yoy_disp[col] = df_yoy_disp[col].apply(lambda x: f"+{x:.1f}%" if pd.notna(x) and x > 0 else (f"{x:.1f}%" if pd.notna(x) else "-"))
        st.dataframe(df_yoy_disp, use_container_width=True, hide_index=True)
    
    # Trend confronto
    st.markdown("##### Trend Mensile Richieste - Confronto Anni")
    anni_trend = st.multiselect("Anni da confrontare", anni_richiesta, default=anni_richiesta[-2:] if len(anni_richiesta) >= 2 else anni_richiesta)
    
    if anni_trend:
        fig_trend = go.Figure()
        for i, anno in enumerate(anni_trend):
            df_a = df[df["ANNO_RICHIESTA"] == anno]
            monthly = df_a.groupby("MESE_RICHIESTA").size().reset_index(name="Pratiche")
            monthly["Mese_Nome"] = monthly["MESE_RICHIESTA"].apply(lambda x: mesi[int(x)-1] if pd.notna(x) and 1 <= x <= 12 else "")
            fig_trend.add_trace(go.Scatter(x=monthly["Mese_Nome"], y=monthly["Pratiche"], name=str(anno), mode='lines+markers', line=dict(width=3, color=COLORS['year_colors'][i % len(COLORS['year_colors'])])))
        fig_trend.update_layout(title="Richieste per Mese", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig_trend, use_container_width=True)

# =============================================================================
# TAB 4: OUTLOOK 2026
# =============================================================================
with tab4:
    st.markdown("### 🔮 Outlook 2026")
    st.caption("Stato delle pratiche per viaggi nel 2026")
    
    df_2026 = df[df["ANNO_VIAGGIO"] == 2026]
    
    if len(df_2026) == 0:
        st.warning("Nessuna pratica con viaggio previsto nel 2026")
    else:
        conv_2026 = df_2026[df_2026["TIPO"] == "CONVERTITA"]
        non_conv_2026 = df_2026[df_2026["TIPO"] == "NON CONVERTITA"]
        lav_2026 = df_2026[df_2026["TIPO"] == "IN LAVORAZIONE"]
        altro_2026 = df_2026[df_2026["TIPO"] == "ALTRO"]
        
        # Pratiche chiuse = Totale - In Lavorazione
        chiuse_2026 = len(df_2026) - len(lav_2026)
        perse_2026 = chiuse_2026 - len(conv_2026)
        cr_2026 = (len(conv_2026) / chiuse_2026 * 100) if chiuse_2026 > 0 else 0
        
        # KPI 2026
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ Confermate 2026", len(conv_2026), fmt_curr(conv_2026["FATTURATO TOT"].sum()))
        col2.metric("🔄 In Lavorazione 2026", len(lav_2026), fmt_curr(lav_2026["FATTURATO TOT"].sum()))
        col3.metric("❌ Perse 2026", perse_2026)
        col4.metric("🎯 CR 2026 (parziale)", f"{cr_2026:.1f}%")
        
        # Pipeline 2026
        st.markdown("##### 📊 Portafoglio Ordini 2026")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Grafico stato
            status_2026 = pd.DataFrame({
                "Stato": ["Confermate", "In Lavorazione", "Perse"],
                "Pratiche": [len(conv_2026), len(lav_2026), len(non_conv_2026)],
                "Valore": [conv_2026["FATTURATO TOT"].sum(), lav_2026["FATTURATO TOT"].sum(), non_conv_2026["FATTURATO TOT"].sum()]
            })
            
            fig = go.Figure(data=[go.Pie(
                labels=status_2026["Stato"], values=status_2026["Valore"], hole=0.5,
                marker=dict(colors=[COLORS['primary'], COLORS['accent'], COLORS['warning']]),
                textinfo='label+percent'
            )])
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), showlegend=False,
                             title="Valore per Stato")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Riepilogo valori
            st.markdown("**Riepilogo Valori 2026**")
            fatt_confermato = conv_2026["FATTURATO TOT"].sum()
            fatt_potenziale = lav_2026["FATTURATO TOT"].sum()
            fatt_perso = non_conv_2026["FATTURATO TOT"].sum()
            
            st.markdown(f"""
            <div class="highlight-box">
                <b>✅ Fatturato Confermato:</b> {fmt_curr(fatt_confermato)}<br>
                <small>{len(conv_2026)} pratiche - {int(conv_2026["N DI PAX"].sum())} PAX</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #1e3a5f; border-radius: 10px; padding: 15px; border-left: 4px solid {COLORS['accent']};">
                <b>🔄 Potenziale In Lavorazione:</b> {fmt_curr(fatt_potenziale)}<br>
                <small>{len(lav_2026)} pratiche da chiudere</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="warning-box">
                <b>❌ Valore Perso:</b> {fmt_curr(fatt_perso)}<br>
                <small>{len(non_conv_2026)} pratiche non convertite</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Distribuzione mensile 2026
        st.markdown("##### Viaggi 2026 per Mese")
        monthly_2026 = df_2026.groupby(["MESE_VIAGGIO", "TIPO"]).agg({"Nome": "count", "FATTURATO TOT": "sum"}).reset_index()
        monthly_2026.columns = ["Mese", "Tipo", "Pratiche", "Fatturato"]
        monthly_2026["Mese_Nome"] = monthly_2026["Mese"].apply(lambda x: mesi[int(x)-1] if pd.notna(x) and 1 <= x <= 12 else "")
        
        # Rinomina tipi per display
        tipo_labels = {"CONVERTITA": "Confermate", "NON CONVERTITA": "Sospese/Ann.", "IN LAVORAZIONE": "In Lavorazione"}
        monthly_2026["Tipo"] = monthly_2026["Tipo"].map(tipo_labels).fillna(monthly_2026["Tipo"])
        
        fig = px.bar(monthly_2026, x="Mese_Nome", y="Fatturato", color="Tipo", barmode="stack",
                     color_discrete_map={"Confermate": COLORS['primary'], "Sospese/Ann.": COLORS['warning'], "In Lavorazione": COLORS['accent']},
                     category_orders={"Mese_Nome": mesi})
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        
        # Lista pratiche in lavorazione
        st.markdown("##### 🔄 Dettaglio Pratiche In Lavorazione 2026")
        if len(lav_2026) > 0:
            lav_display = lav_2026[["Nome", "CLIENTE", "PAESE", "DATA VIAGGIO", "FATTURATO TOT", "N DI PAX", "ASSEGNATO"]].copy()
            lav_display["FATTURATO TOT"] = lav_display["FATTURATO TOT"].apply(lambda x: f"€{x:,.0f}")
            lav_display["DATA VIAGGIO"] = lav_display["DATA VIAGGIO"].dt.strftime("%d/%m/%Y").fillna("-")
            st.dataframe(lav_display.sort_values("DATA VIAGGIO"), use_container_width=True, hide_index=True)
        
        # Confermate 2026
        st.markdown("##### ✅ Pratiche Confermate 2026")
        if len(conv_2026) > 0:
            conv_display = conv_2026[["Nome", "CLIENTE", "PAESE", "DATA VIAGGIO", "FATTURATO TOT", "N DI PAX"]].copy()
            conv_display["FATTURATO TOT"] = conv_display["FATTURATO TOT"].apply(lambda x: f"€{x:,.0f}")
            conv_display["DATA VIAGGIO"] = conv_display["DATA VIAGGIO"].dt.strftime("%d/%m/%Y").fillna("-")
            st.dataframe(conv_display.sort_values("DATA VIAGGIO"), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: MERCATI
# =============================================================================
with tab5:
    st.markdown("### 🌍 Analisi Mercati e Clienti")
    
    # Filtro anno
    anno_mercati = st.selectbox("Anno viaggio", ["Tutti"] + [str(a) for a in anni_viaggio], index=0, key="mercati_anno")
    
    if anno_mercati == "Tutti":
        df_merc = df_convertite
    else:
        df_merc = df_convertite[df_convertite["ANNO_VIAGGIO"] == int(anno_mercati)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Top 10 Paesi")
        paese_stats = df_merc.groupby("PAESE").agg({"FATTURATO TOT": "sum", "MARGINE LORDO": "sum", "Nome": "count"}).reset_index()
        paese_stats.columns = ["Paese", "Fatturato", "Margine", "Pratiche"]
        paese_stats["Margine %"] = (paese_stats["Margine"] / paese_stats["Fatturato"] * 100).round(1)
        paese_stats = paese_stats.sort_values("Fatturato", ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(y=paese_stats["Paese"], x=paese_stats["Fatturato"], orientation='h',
            marker=dict(color=paese_stats["Margine %"], colorscale=[[0, COLORS['warning']], [1, COLORS['primary']]], showscale=True, colorbar=dict(title="Marg%")),
            text=paese_stats["Fatturato"].apply(fmt_curr), textposition='outside'))
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Top 10 Clienti")
        cliente_stats = df_merc.groupby("CLIENTE").agg({"FATTURATO TOT": "sum", "MARGINE LORDO": "sum", "Nome": "count"}).reset_index()
        cliente_stats.columns = ["Cliente", "Fatturato", "Margine", "Pratiche"]
        cliente_stats["Margine %"] = (cliente_stats["Margine"] / cliente_stats["Fatturato"] * 100).round(1)
        cliente_stats = cliente_stats.sort_values("Fatturato", ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(y=cliente_stats["Cliente"], x=cliente_stats["Fatturato"], orientation='h',
            marker=dict(color=cliente_stats["Margine %"], colorscale=[[0, COLORS['warning']], [1, COLORS['primary']]], showscale=True, colorbar=dict(title="Marg%")),
            text=cliente_stats["Fatturato"].apply(fmt_curr), textposition='outside'))
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 6: TEAM (discreto)
# =============================================================================
with tab6:
    st.markdown("### 👥 Analisi Team")
    st.caption("Performance del team - focus su pratiche gestite e conversion rate")
    
    # Filtro anno
    anno_team = st.selectbox("Anno viaggio", ["Tutti"] + [str(a) for a in anni_viaggio], index=0, key="team_anno")
    
    if anno_team == "Tutti":
        df_team_filtered = df
    else:
        df_team_filtered = df[df["ANNO_VIAGGIO"] == int(anno_team)]
    
    assegnati = [a for a in df_team_filtered["ASSEGNATO"].dropna().unique() if a and str(a).strip() and 'EX ' not in str(a)]
    
    team_stats = []
    for ass in assegnati:
        df_ass = df_team_filtered[df_team_filtered["ASSEGNATO"] == ass]
        conv = df_ass[df_ass["TIPO"] == "CONVERTITA"]
        lav = df_ass[df_ass["TIPO"] == "IN LAVORAZIONE"]
        # Pratiche chiuse = Totale - In Lavorazione
        chiuse = len(df_ass) - len(lav)
        non_conv_count = chiuse - len(conv)
        
        team_stats.append({
            "Assegnato": ass,
            "Pratiche Gestite": len(df_ass),
            "Convertite": len(conv),
            "Non Conv.": non_conv_count,
            "CR %": (len(conv) / chiuse * 100) if chiuse > 0 else 0,
            "Fatturato": conv["FATTURATO TOT"].sum(),
            "Margine %": (conv["MARGINE LORDO"].sum() / conv["FATTURATO TOT"].sum() * 100) if conv["FATTURATO TOT"].sum() > 0 else 0
        })
    
    df_team = pd.DataFrame(team_stats).sort_values("Pratiche Gestite", ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Pratiche Gestite e CR")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df_team["Assegnato"], y=df_team["Pratiche Gestite"], name="Pratiche", marker_color=COLORS['secondary']), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_team["Assegnato"], y=df_team["CR %"], name="CR %", mode="lines+markers", line=dict(color=COLORS['primary'], width=3)), secondary_y=True)
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), legend=dict(orientation='h', y=1.1), yaxis2=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Riepilogo")
        df_team_disp = df_team[["Assegnato", "Pratiche Gestite", "Convertite", "Non Conv.", "CR %", "Margine %"]].copy()
        df_team_disp["CR %"] = df_team_disp["CR %"].round(1)
        df_team_disp["Margine %"] = df_team_disp["Margine %"].round(1)
        st.dataframe(df_team_disp, use_container_width=True, hide_index=True)
    
    with st.expander("📊 Dettaglio Fatturato"):
        df_fatt = df_team[["Assegnato", "Fatturato", "Convertite"]].copy()
        df_fatt["Fatt. Medio"] = (df_fatt["Fatturato"] / df_fatt["Convertite"]).fillna(0)
        df_fatt["Fatturato"] = df_fatt["Fatturato"].apply(fmt_curr)
        df_fatt["Fatt. Medio"] = df_fatt["Fatt. Medio"].apply(fmt_curr)
        st.dataframe(df_fatt, use_container_width=True, hide_index=True)

# --- FOOTER ---
st.markdown("---")
st.caption(f"Dashboard aggiornata al {datetime.now().strftime('%d/%m/%Y %H:%M')} | {len(df)} pratiche totali")
