import streamlit as st
import matplotlib.pyplot as plt
from utils.config import CORES
from utils.helpers import nova_fig, estilo_ax, bloco


def render(df_diario, df_mensal):
    st.markdown(f"""
    <div class='hero'>
        <h1>Forecasting Climático</h1>
        <p>Uma jornada completa de Machine Learning com Séries Temporais —
        da análise exploratória à previsão do futuro com ARIMA, SARIMA e Prophet.</p>
        <br>
        <span class='badge'>ARIMA</span>
        <span class='badge'>SARIMA</span>
        <span class='badge'>Prophet</span>
        <span class='badge'>Séries Temporais</span>
        <span class='badge'>AVD & Gestalt</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    anos = (df_mensal.index[-1] - df_mensal.index[0]).days // 365
    for col, titulo, valor, desc in [
        (c1, "Anos de dados",  f"{anos}",                         "Período histórico"),
        (c2, "Temp. Média",    f"{df_mensal.mean():.1f}°C",       "Média mensal histórica"),
        (c3, "Máxima",         f"{df_diario['temperatura'].max():.1f}°C", "Pico registrado"),
        (c4, "Mínima",         f"{df_diario['temperatura'].min():.1f}°C", "Menor registrada"),
    ]:
        col.markdown(
            f"<div class='card-metrica'>"
            f"<h3>{titulo}</h3>"
            f"<div class='valor'>{valor}</div>"
            f"<div class='desc'>{desc}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    fig, ax = nova_fig(14, 3.5)
    ax.fill_between(df_mensal.index, df_mensal.values, df_mensal.min(),
                    alpha=0.12, color=CORES["primaria"])
    ax.plot(df_mensal.index, df_mensal.values, color=CORES["primaria"], linewidth=1.2)
    estilo_ax(ax, "Temperatura Média Mensal — Visão Geral do Período")
    ax.set_ylabel("°C")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

