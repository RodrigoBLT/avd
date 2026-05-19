

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from utils.config import CSS
from utils.data import carregar_dados, split_treino_teste

# Importa páginas
from modulos import pg_inicio, pg_eda, pg_preprocessamento
from modulos import pg_modelos, pg_forecasting, pg_conclusao, pg_dashboard

st.set_page_config(
    page_title="Forecasting Climático — Séries Temporais",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

PAGINAS = {
    "Início":                      "inicio",
    "Análise Exploratória (EDA)":  "eda",
    "Pré-processamento":           "preprocessamento",
    "Treinamento dos Modelos":     "modelos",
    "Forecasting":                 "forecasting",
    "Conclusão & Storytelling":    "conclusao",
    "Dashboard Interativo":        "dashboard",
}

with st.sidebar:
    st.markdown("<div class='sidebar-titulo'>Forecasting Climático</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    pagina = st.radio("Navegação", list(PAGINAS.keys()), label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-info'>
        Seminário — Tema 6<br>
        Séries Temporais e Forecasting<br>
        Dataset: Weather History (Kaggle)
    </div>
    """, unsafe_allow_html=True)

try:
    df_diario, df_mensal = carregar_dados()
    treino, teste = split_treino_teste(df_mensal)

    rota = PAGINAS[pagina]

    if rota == "inicio":
        pg_inicio.render(df_diario, df_mensal)

    elif rota == "eda":
        pg_eda.render(df_diario, df_mensal)

    elif rota == "preprocessamento":
        pg_preprocessamento.render(df_diario, df_mensal)

    elif rota == "modelos":
        pg_modelos.render(treino, teste)

    elif rota == "forecasting":
        pg_forecasting.render(df_mensal)

    elif rota == "conclusao":
        pg_conclusao.render()

    elif rota == "dashboard":
        pg_dashboard.render(df_diario, df_mensal, treino, teste)

except FileNotFoundError:
    st.error("❌ Arquivo 'weatherHistory.csv' não encontrado!")
    st.markdown("""
    **Para continuar:**
    1. Acesse: https://www.kaggle.com/datasets/muthuj7/weather-dataset
    2. Baixe o arquivo `weatherHistory.csv`
    3. Coloque na **mesma pasta** deste script (`app.py`)
    4. Execute: `streamlit run app.py`
    """)
