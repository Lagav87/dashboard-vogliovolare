import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    layout="wide", 
    page_title="VoglioVolare BI Dashboard",
    page_icon="✈️"
)

# --- STILE CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    .stApp { font-family: 'DM Sans', sans-serif; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00d4aa;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
    }
    
    .year-selector {
        background: #1e3a5f;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .insight-box {
        background: #0d2137;
        border-left: 4px solid #00d4aa;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- COLORI ---
COLORS = {
    'primary': '#00d4aa',
    'secondary': '#1e3a5f',
    'accent': '#e94560',
    'warning': '#ff6b6b',
    'text': '#ccd6f6',
    'muted': '#8892b0'
}

# --- MAPPATURA COLONNE MONDAY ---
COLUMN_MAPPING = {
    "cliente": "dup__of_note", 
    "paese": "dup__of_cliente", 
    "responsabile": "person", 
    "assegnato": "stato",
    "status": "status7", 
    "arrivo_richiesta": "dup__of_last_update",
    "invio_offerta": "data",
    "date_viaggio": "timeline7", 
    "n_pax": "numeri2", 
    "fatturato_tot": "numbers4", 
    "costo_tot": "numbers1", 
    "margine_lordo": "formula",
}

# --- FUNZIONI ---
def fetch_monday_data(api_key, board_id):
    url = "https://api.monday.com/v2"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    all_items = []
    cursor = None
    
    while True:
        if cursor:
            query = f'''query {{ boards (ids: {board_id}) {{ items_page (limit: 500, cursor: "{cursor}") {{ cursor items {{ name group {{ title }} column_values {{ id text value }} }} }} }} }}'''
        else:
            query = f'''query {{ boards (ids: {board_id}) {{ items_page (limit: 500) {{ cursor items {{ name group {{ title }} column_values {{ id text value }} }} }} }} }}'''
        
        try:
            response = requests.post(url, json={'query': query}, headers=headers)
            if response.status_code == 200:
                data = response.json()
                items_page = data['data']['boards'][0]['items_page']
                all_items.extend(items_page['items'])
                cursor = items_page.get('cursor')
                if not cursor:
                    break
            else:
                return None
        except:
            return None
    
    return {'data': {'boards': [{'items_page': {'items': all_items}}]}}

def parse_currency(value):
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    clean_str = str(value).replace("€", "").replace("$", "").replace("£", "").strip()
    if not clean_str:
        return 0.0
    try:
        if "," in clean_str and "." in clean_str:
            if clean_str.find(".") < clean_str.find(","):
                clean_str = clean_str.replace(".", "").replace(",", ".")
        elif "," in clean_str:
            clean_str = clean_str.replace(",", ".")
        return float(clean_str)
    except:
        return 0.0

def process_monday_data(raw_data):
    try:
        items = raw_data['data']['boards'][0]['items_page']['items']
    except:
        return pd.DataFrame()
    
    processed_rows = []
    
    for item in items:
        row = {}
        row["Nome"] = item['name']
        
        try:
            row["GRUPPO"] = item['group']['title']
        except:
            row["GRUPPO"] = "N/A"
        
        # Mappa colonne
        col_map_text = {c['id']: c['text'] for c in item['column_values']}
        col_map_value = {}
        for c in item['column_values']:
            if c.get('value'):
                try:
                    val = json.loads(c['value'])
                    if isinstance(val, dict) and 'value' in val:
                        col_map_value[c['id']] = val['value']
                    elif isinstance(val, (int, float)):
                        col_map_value[c['id']] = val
                except:
                    pass
        
        row["CLIENTE"] = col_map_text.get(COLUMN_MAPPING["cliente"], "")
        row["PAESE"] = col_map_text.get(COLUMN_MAPPING["paese"], "")
        row["ASSEGNATO"] = col_map_text.get(COLUMN_MAPPING["assegnato"], "")
        row["STATUS"] = col_map_text.get(COLUMN_MAPPING["status"], "")
        row["ARRIVO RICHIESTA"] = col_map_text.get(COLUMN_MAPPING["arrivo_richiesta"], "")
        row["INVIO OFFERTA"] = col_map_text.get(COLUMN_MAPPING["invio_offerta"], "")
        
        # Date viaggio
        raw_viaggio = col_map_text.get(COLUMN_MAPPING["date_viaggio"], "")
        if raw_viaggio and " - " in raw_viaggio:
            row["DATE VIAGGIO START"] = raw_viaggio.split(" - ")[0]
        else:
            row["DATE VIAGGIO START"] = raw_viaggio
        
        # Numeri
        row["N DI PAX"] = parse_currency(col_map_value.get(COLUMN_MAPPING["n_pax"]) or col_map_text.get(COLUMN_MAPPING["n_pax"], 0))
        row["FATTURATO TOT"] = parse_currency(col_map_value.get(COLUMN_MAPPING["fatturato_tot"]) or col_map_text.get(COLUMN_MAPPING["fatturato_tot"], 0))
        row["COSTO TOT"] = parse_currency(col_map_value.get(COLUMN_MAPPING["costo_tot"]) or col_map_text.get(COLUMN_MAPPING["costo_tot"], 0))
        row["MARGINE LORDO"] = parse_currency(col_map_value.get(COLUMN_MAPPING["margine_lordo"]) or col_map_text.get(COLUMN_MAPPING["margine_lordo"], 0))
        
        # Calcola margine se mancante
        if row["MARGINE LORDO"] == 0 and row["FATTURATO TOT"] > 0 and row["COSTO TOT"] > 0:
            row["MARGINE LORDO"] = row["FATTURATO TOT"] - row["COSTO TOT"]
        
        row["MARGINE %"] = (row["MARGINE LORDO"] / row["FATTURATO TOT"] * 100) if row["FATTURATO TOT"] > 0 else 0
        
        processed_rows.append(row)
    
    df = pd.DataFrame(processed_rows)
    
    # Date
    df["DATE VIAGGIO START"] = pd.to_datetime(df["DATE VIAGGIO START"], errors='coerce')
    df["ARRIVO RICHIESTA"] = pd.to_datetime(df["ARRIVO RICHIESTA"], errors='coerce')
    df["INVIO OFFERTA"] = pd.to_datetime(df["INVIO OFFERTA"], errors='coerce')
    
    # Anno viaggio
    df["ANNO"] = df["DATE VIAGGIO START"].dt.year
    df["MESE"] = df["DATE VIAGGIO START"].dt.month
    
    # Tipo pratica dal gruppo
    def classifica_tipo(gruppo):
        g = str(gruppo).upper()
        if "REALIZZAT" in g:
            return "REALIZZATO"
        elif "CONFERMAT" in g:
            return "CONFERMATO"
        elif "SOSPES" in g:
            return "SOSPESO"
        elif "LAVORAZIONE" in g:
            return "IN LAVORAZIONE"
        return "ALTRO"
    
    df["TIPO"] = df["GRUPPO"].apply(classifica_tipo)
    
    # Giorni quotazione
    df["GIORNI_QUOTAZIONE"] = (df["INVIO OFFERTA"] - df["ARRIVO RICHIESTA"]).dt.days
    
    return df

def format_currency(value):
    if value >= 1000000:
        return f"€{value/1000000:.2f}M"
    elif value >= 1000:
        return f"€{value/1000:.0f}K"
    return f"€{value:.0f}"

# --- INTERFACCIA ---
st.markdown("# ✈️ VoglioVolare BI Dashboard")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Connessione Monday")
    monday_api_key = st.text_input("API Key", type="password")
    board_id = st.text_input("Board ID")
    load_btn = st.button("🚀 Carica Dati", type="primary", use_container_width=True)

if "df_monday" not in st.session_state:
    st.session_state["df_monday"] = None

if load_btn and monday_api_key and board_id:
    with st.spinner("Caricamento..."):
        raw_data = fetch_monday_data(monday_api_key, board_id)
        if raw_data:
            df = process_monday_data(raw_data)
            if len(df) > 0:
                st.session_state["df_monday"] = df
                st.success(f"✅ {len(df)} pratiche caricate")
                st.rerun()

if st.session_state["df_monday"] is None:
    st.info("👈 Inserisci API Key e Board ID nella sidebar per iniziare")
    st.stop()

df = st.session_state["df_monday"]

# --- FILTRO ANNO PRINCIPALE ---
anni_disponibili = sorted([int(a) for a in df["ANNO"].dropna().unique() if a >= 2020])

st.markdown("### 📅 Seleziona Anno")
col_anni = st.columns(len(anni_disponibili) + 1)

with col_anni[0]:
    tutti_anni = st.checkbox("Tutti", value=True)

anno_selezionato = []
if tutti_anni:
    anno_selezionato = anni_disponibili
else:
    for i, anno in enumerate(anni_disponibili):
        with col_anni[i + 1]:
            if st.checkbox(str(anno), value=False):
                anno_selezionato.append(anno)

if not anno_selezionato:
    anno_selezionato = anni_disponibili

# Filtra per anno
df_anno = df[df["ANNO"].isin(anno_selezionato)]

# --- KPI GENERALI ---
st.markdown("---")
st.markdown("## 📊 Riepilogo Generale")

df_realizzati = df_anno[df_anno["TIPO"] == "REALIZZATO"]
df_sospesi = df_anno[df_anno["TIPO"] == "SOSPESO"]
df_confermati = df_anno[df_anno["TIPO"] == "CONFERMATO"]
df_lavorazione = df_anno[df_anno["TIPO"] == "IN LAVORAZIONE"]

fatturato = df_realizzati["FATTURATO TOT"].sum()
margine = df_realizzati["MARGINE LORDO"].sum()
margine_perc = (margine / fatturato * 100) if fatturato > 0 else 0
pax = df_realizzati["N DI PAX"].sum()
pratiche_chiuse = len(df_realizzati) + len(df_sospesi)
conversion = (len(df_realizzati) / pratiche_chiuse * 100) if pratiche_chiuse > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Fatturato", format_currency(fatturato))
col2.metric("📈 Margine", format_currency(margine), f"{margine_perc:.1f}%")
col3.metric("🎯 Conversion Rate", f"{conversion:.1f}%")
col4.metric("👥 PAX", f"{int(pax):,}")
col5.metric("📋 Pratiche Realizzate", len(df_realizzati))

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Andamento Annuale",
    "🌍 Mercati & Clienti",
    "👥 Performance Team",
    "📋 Dati"
])

# =============================================================================
# TAB 1: ANDAMENTO ANNUALE
# =============================================================================
with tab1:
    st.markdown("### 📈 Performance per Anno")
    
    # Calcolo metriche per anno
    anni_stats = []
    for anno in sorted(df["ANNO"].dropna().unique()):
        if anno < 2020:
            continue
        df_a = df[df["ANNO"] == anno]
        real = df_a[df_a["TIPO"] == "REALIZZATO"]
        sosp = df_a[df_a["TIPO"] == "SOSPESO"]
        conf = df_a[df_a["TIPO"] == "CONFERMATO"]
        lav = df_a[df_a["TIPO"] == "IN LAVORAZIONE"]
        
        fatt = real["FATTURATO TOT"].sum()
        marg = real["MARGINE LORDO"].sum()
        chiuse = len(real) + len(sosp)
        conv = (len(real) / chiuse * 100) if chiuse > 0 else 0
        
        anni_stats.append({
            "Anno": int(anno),
            "Fatturato": fatt,
            "Margine": marg,
            "Margine %": (marg / fatt * 100) if fatt > 0 else 0,
            "PAX": real["N DI PAX"].sum(),
            "Realizzate": len(real),
            "Sospese": len(sosp),
            "Confermate": len(conf),
            "In Lavorazione": len(lav),
            "Conversion %": conv
        })
    
    df_anni = pd.DataFrame(anni_stats)
    
    if len(df_anni) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Fatturato e Margine per Anno")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df_anni["Anno"].astype(str), y=df_anni["Fatturato"], name="Fatturato", marker_color=COLORS['primary']), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_anni["Anno"].astype(str), y=df_anni["Margine %"], name="Margine %", mode="lines+markers", line=dict(color=COLORS['accent'], width=3), marker=dict(size=10)), secondary_y=True)
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), legend=dict(orientation='h', y=1.1), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), yaxis2=dict(showgrid=False, range=[0, 60]))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Conversion Rate per Anno")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_anni["Anno"].astype(str), y=df_anni["Conversion %"], marker_color=df_anni["Conversion %"], marker_colorscale=[[0, COLORS['warning']], [0.5, '#f39c12'], [1, COLORS['primary']]], text=df_anni["Conversion %"].round(1).astype(str) + "%", textposition='outside'))
            fig2.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 100]))
            st.plotly_chart(fig2, use_container_width=True)
        
        # Tabella riepilogo anni
        st.markdown("##### Riepilogo per Anno")
        df_display = df_anni.copy()
        df_display["Fatturato"] = df_display["Fatturato"].apply(lambda x: f"€{x:,.0f}")
        df_display["Margine"] = df_display["Margine"].apply(lambda x: f"€{x:,.0f}")
        df_display["Margine %"] = df_display["Margine %"].round(1)
        df_display["Conversion %"] = df_display["Conversion %"].round(1)
        df_display["PAX"] = df_display["PAX"].astype(int)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: MERCATI & CLIENTI
# =============================================================================
with tab2:
    st.markdown("### 🌍 Analisi Mercati e Clienti")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Top 10 Paesi per Fatturato")
        paese_stats = df_realizzati.groupby("PAESE").agg({
            "FATTURATO TOT": "sum",
            "MARGINE LORDO": "sum",
            "N DI PAX": "sum",
            "Nome": "count"
        }).reset_index()
        paese_stats.columns = ["Paese", "Fatturato", "Margine", "PAX", "Pratiche"]
        paese_stats["Margine %"] = (paese_stats["Margine"] / paese_stats["Fatturato"] * 100).round(1)
        paese_stats = paese_stats.sort_values("Fatturato", ascending=True).tail(10)
        
        if len(paese_stats) > 0:
            fig = go.Figure(go.Bar(
                y=paese_stats["Paese"],
                x=paese_stats["Fatturato"],
                orientation='h',
                marker=dict(color=paese_stats["Margine %"], colorscale=[[0, COLORS['warning']], [0.5, '#f39c12'], [1, COLORS['primary']]], showscale=True, colorbar=dict(title="Marg%")),
                text=paese_stats["Fatturato"].apply(format_currency),
                textposition='outside'
            ))
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Top 10 Clienti per Fatturato")
        cliente_stats = df_realizzati.groupby("CLIENTE").agg({
            "FATTURATO TOT": "sum",
            "MARGINE LORDO": "sum",
            "Nome": "count"
        }).reset_index()
        cliente_stats.columns = ["Cliente", "Fatturato", "Margine", "Pratiche"]
        cliente_stats["Margine %"] = (cliente_stats["Margine"] / cliente_stats["Fatturato"] * 100).round(1)
        cliente_stats = cliente_stats.sort_values("Fatturato", ascending=True).tail(10)
        
        if len(cliente_stats) > 0:
            fig = go.Figure(go.Bar(
                y=cliente_stats["Cliente"],
                x=cliente_stats["Fatturato"],
                orientation='h',
                marker=dict(color=cliente_stats["Margine %"], colorscale=[[0, COLORS['warning']], [0.5, '#f39c12'], [1, COLORS['primary']]], showscale=True, colorbar=dict(title="Marg%")),
                text=cliente_stats["Fatturato"].apply(format_currency),
                textposition='outside'
            ))
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: PERFORMANCE TEAM
# =============================================================================
with tab3:
    st.markdown("### 👥 Performance per Assegnato")
    
    # Prima mostra i dati generali, poi il dettaglio per assegnato
    assegnati = sorted([a for a in df_anno["ASSEGNATO"].dropna().unique() if a and str(a).strip() != '' and 'EX ' not in str(a)])
    
    # Calcolo stats per assegnato
    team_stats = []
    for ass in assegnati:
        df_ass = df_anno[df_anno["ASSEGNATO"] == ass]
        real = df_ass[df_ass["TIPO"] == "REALIZZATO"]
        sosp = df_ass[df_ass["TIPO"] == "SOSPESO"]
        
        fatt = real["FATTURATO TOT"].sum()
        marg = real["MARGINE LORDO"].sum()
        chiuse = len(real) + len(sosp)
        conv = (len(real) / chiuse * 100) if chiuse > 0 else 0
        
        team_stats.append({
            "Assegnato": ass,
            "Fatturato": fatt,
            "Margine": marg,
            "Margine %": (marg / fatt * 100) if fatt > 0 else 0,
            "PAX": real["N DI PAX"].sum(),
            "Realizzate": len(real),
            "Sospese": len(sosp),
            "Conversion %": conv
        })
    
    df_team = pd.DataFrame(team_stats)
    df_team = df_team.sort_values("Fatturato", ascending=False)
    
    if len(df_team) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Fatturato e Margine % per Assegnato")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df_team["Assegnato"], y=df_team["Fatturato"], name="Fatturato", marker_color=COLORS['primary'], text=df_team["Fatturato"].apply(format_currency), textposition='outside'), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_team["Assegnato"], y=df_team["Margine %"], name="Margine %", mode="lines+markers", line=dict(color=COLORS['accent'], width=3), marker=dict(size=10)), secondary_y=True)
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), legend=dict(orientation='h', y=1.15), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), yaxis2=dict(showgrid=False, range=[0, 50]))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Conversion Rate per Assegnato")
            df_team_sorted = df_team.sort_values("Conversion %", ascending=True)
            fig2 = go.Figure(go.Bar(
                y=df_team_sorted["Assegnato"],
                x=df_team_sorted["Conversion %"],
                orientation='h',
                marker=dict(color=df_team_sorted["Conversion %"], colorscale=[[0, COLORS['warning']], [0.5, '#f39c12'], [1, COLORS['primary']]]),
                text=df_team_sorted["Conversion %"].round(1).astype(str) + "%",
                textposition='outside'
            ))
            fig2.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['text']), xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 100]))
            st.plotly_chart(fig2, use_container_width=True)
        
        # Tabella team
        st.markdown("##### Riepilogo Team")
        df_team_display = df_team.copy()
        df_team_display["Fatturato"] = df_team_display["Fatturato"].apply(lambda x: f"€{x:,.0f}")
        df_team_display["Margine"] = df_team_display["Margine"].apply(lambda x: f"€{x:,.0f}")
        df_team_display["Margine %"] = df_team_display["Margine %"].round(1)
        df_team_display["Conversion %"] = df_team_display["Conversion %"].round(1)
        df_team_display["PAX"] = df_team_display["PAX"].astype(int)
        st.dataframe(df_team_display, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: DATI
# =============================================================================
with tab4:
    st.markdown("### 📋 Dati Dettaglio")
    
    # Filtri
    col1, col2, col3 = st.columns(3)
    with col1:
        tipo_filter = st.multiselect("Tipo", df_anno["TIPO"].unique(), default=df_anno["TIPO"].unique())
    with col2:
        paese_filter = st.multiselect("Paese", sorted(df_anno["PAESE"].dropna().unique()))
    with col3:
        ass_filter = st.multiselect("Assegnato", assegnati)
    
    df_view = df_anno[df_anno["TIPO"].isin(tipo_filter)]
    if paese_filter:
        df_view = df_view[df_view["PAESE"].isin(paese_filter)]
    if ass_filter:
        df_view = df_view[df_view["ASSEGNATO"].isin(ass_filter)]
    
    cols = ["Nome", "CLIENTE", "PAESE", "ASSEGNATO", "STATUS", "TIPO", "ANNO", "N DI PAX", "FATTURATO TOT", "MARGINE LORDO", "MARGINE %"]
    cols = [c for c in cols if c in df_view.columns]
    
    df_display = df_view[cols].copy()
    df_display["FATTURATO TOT"] = df_display["FATTURATO TOT"].apply(lambda x: f"€{x:,.0f}")
    df_display["MARGINE LORDO"] = df_display["MARGINE LORDO"].apply(lambda x: f"€{x:,.0f}")
    df_display["MARGINE %"] = df_display["MARGINE %"].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)
    
    st.download_button("📥 Scarica CSV", df_anno.to_csv(index=False).encode('utf-8'), f"export_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# Footer
st.markdown("---")
st.caption(f"Dati aggiornati al {datetime.now().strftime('%d/%m/%Y %H:%M')} | {len(df_anno)} pratiche")