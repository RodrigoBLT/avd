"""
Configurações globais: paleta de cores, layout Plotly/Matplotlib e CSS do app.
"""

CORES = {
    "primaria":   "#00c8ff",
    "secundaria": "#ffaa00",
    "terciaria":  "#bf7fff",
    "real":       "#ff6b6b",
    "ok":         "#00ff99",
    "fundo":      "#0f1117",
    "card":       "#1a1d27",
    "grid":       "#1e2130",
    "texto":      "#aaaaaa",
}

# Cores por modelo (usadas em todos os gráficos para consistência — Gestalt: Similaridade)
CORES_MODELO = {
    "ARIMA":   "#00c8ff",
    "SARIMA":  "#ffaa00",
    "Prophet": "#bf7fff",
    "Real":    "#ff6b6b",
    "Treino":  "#444455",
}

# Layout base reutilizável para figuras Plotly
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CORES["fundo"],
    plot_bgcolor=CORES["fundo"],
    font=dict(color=CORES["texto"], family="Space Grotesk, sans-serif", size=12),
    xaxis=dict(gridcolor=CORES["grid"], zeroline=False),
    yaxis=dict(gridcolor=CORES["grid"], zeroline=False),
    legend=dict(
        bgcolor=CORES["card"],
        bordercolor="#2a2d3a",
        borderwidth=1,
        font=dict(color="white"),
    ),
    margin=dict(l=50, r=30, t=55, b=40),
    hovermode="x unified",
)

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif;
        background-color: {CORES['fundo']};
        color: #e0e0e0;
    }}
    .main {{ background-color: {CORES['fundo']}; }}
    h1, h2, h3 {{ color: {CORES['primaria']}; }}

    /* ── Cards de métricas ── */
    .card-metrica {{
        background: {CORES['card']};
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 18px 12px;
        text-align: center;
    }}
    .card-metrica h3  {{ color: {CORES['primaria']}; margin: 0 0 4px 0; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .05em; }}
    .card-metrica .valor {{ font-size: 2rem; font-weight: 700; color: white; line-height: 1.1; }}
    .card-metrica .desc  {{ font-size: 0.7rem; color: #666; margin-top: 5px; }}

    /* ── Blocos de destaque ── */
    .bloco {{
        background: {CORES['card']};
        border-left: 3px solid {CORES['primaria']};
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 14px;
        color: #cccccc;
        font-size: 0.93rem;
        line-height: 1.6;
    }}
    .bloco-ok    {{ background: {CORES['card']}; border-left: 3px solid {CORES['ok']};        padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 14px; color: #cccccc; font-size: 0.93rem; line-height: 1.6; }}
    .bloco-aviso {{ background: {CORES['card']}; border-left: 3px solid {CORES['secundaria']}; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 14px; color: #cccccc; font-size: 0.93rem; line-height: 1.6; }}

    /* ── Hero banner ── */
    .hero {{
        background: linear-gradient(135deg, #0d1b2a 0%, {CORES['card']} 60%, {CORES['fundo']} 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 36px 40px;
        margin-bottom: 28px;
        text-align: center;
    }}
    .hero h1 {{ font-size: 2.2rem; color: {CORES['primaria']}; margin-bottom: 6px; }}
    .hero p   {{ color: #999; font-size: 1rem; max-width: 680px; margin: 0 auto; }}

    /* ── Badges ── */
    .badge {{
        display: inline-block;
        background: #1e3a5f22;
        color: {CORES['primaria']};
        border: 1px solid {CORES['primaria']}55;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 3px;
    }}

    /* ── Sidebar ── */
    .sidebar-titulo {{
        color: {CORES['primaria']};
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: .03em;
    }}
    .sidebar-info {{
        color: #444;
        font-size: 0.72rem;
        line-height: 1.6;
    }}

    /* ── Step tracker (storytelling) ── */
    .step-row {{
        display: flex;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 14px;
    }}
    .step-num {{
        min-width: 36px; height: 36px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.9rem;
    }}
    .step-body {{
        background: {CORES['card']};
        border-radius: 8px;
        padding: 12px 16px;
        flex: 1;
        font-size: 0.9rem;
        color: #cccccc;
    }}

    /* ── Botões ── */
    .stButton > button {{
        background: linear-gradient(90deg, {CORES['primaria']}, #0080ff);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 9px 22px; width: 100%;
        transition: opacity .2s;
    }}
    .stButton > button:hover {{ opacity: .88; }}

    /* ── Divider ── */
    hr {{ border-color: #1e2130 !important; margin: 20px 0 !important; }}
</style>
"""
