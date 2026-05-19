import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from utils.config import CORES
from utils.helpers import nova_fig, estilo_ax, bloco, secao


def render(df_diario, df_mensal):
    st.title("Análise Exploratória de Dados (EDA)")

    bloco(
        "<b>O que estamos prevendo?</b><br><br>"
        "Empresas de energia, logística e agronegócio perdem recursos por não antecipar "
        "variações climáticas. Nosso objetivo: construir um modelo capaz de "
        "<b>prever a temperatura dos próximos meses</b> com ARIMA, SARIMA e Prophet."
    )

    c1, c2, c3 = st.columns(3)
    for col, titulo, texto in [
        (c1, "Problema",   "Prever temperatura futura para apoio a decisões operacionais."),
        (c2, "Tipo de ML", "Aprendizado Supervisionado — Regressão Temporal."),
        (c3, "Algoritmos", "ARIMA, SARIMA e Facebook Prophet."),
    ]:
        col.markdown(
            f"<div class='bloco'><b>{titulo}</b><br><span style='font-size:.88rem'>{texto}</span></div>",
            unsafe_allow_html=True,
        )

    secao("Série Temporal Histórica")
    fig, ax = nova_fig(14, 4)
    ax.plot(df_diario.index, df_diario["temperatura"],
            color=CORES["primaria"], linewidth=0.7, alpha=0.9)
    estilo_ax(ax, "Temperatura Diária ao Longo do Tempo")
    ax.set_ylabel("°C")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    bloco(
        "<b>Interpretação:</b> Oscilações cíclicas regulares confirmam forte "
        "<b>sazonalidade anual</b> — ideal para modelos de séries temporais."
    )

    secao("Sazonalidade Mensal")
    df_mes = df_diario.copy()
    df_mes["mes"] = df_mes.index.month
    meses = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor(CORES["fundo"])
    ax = axes[0]
    ax.set_facecolor(CORES["fundo"])
    dados_mes = [df_mes[df_mes["mes"] == m]["temperatura"].values for m in range(1, 13)]
    bp = ax.boxplot(dados_mes, labels=meses, patch_artist=True,
                    medianprops=dict(color=CORES["real"], linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#1a3a4a")
        patch.set_edgecolor(CORES["primaria"])
    estilo_ax(ax, "Distribuição por Mês")

    ax2 = axes[1]
    ax2.set_facecolor(CORES["fundo"])
    media_mes = df_mes.groupby("mes")["temperatura"].mean()
    bars = ax2.bar(meses, media_mes.values, color=CORES["primaria"], alpha=0.8,
                   edgecolor=CORES["fundo"])
    for bar, val in zip(bars, media_mes.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}°", ha="center", color="white", fontsize=8)
    estilo_ax(ax2, "Temperatura Média por Mês")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    secao("Correlação entre Variáveis")
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(CORES["fundo"])
    ax.set_facecolor(CORES["fundo"])
    corr = df_diario[["temperatura", "umidade", "vento"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, linecolor=CORES["fundo"],
                annot_kws={"color": "white"})
    estilo_ax(ax, "Mapa de Correlação")
    st.pyplot(fig)
    plt.close()

    secao("Teste de Estacionaridade (ADF)")
    resultado = adfuller(df_mensal.dropna())
    pvalue, stat = resultado[1], resultado[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Estatística ADF", f"{stat:.4f}")
    c2.metric("p-value", f"{pvalue:.6f}")
    c3.metric("Conclusão", "Estacionária" if pvalue < 0.05 else "Não Estacionária")

    tipo = "ok" if pvalue < 0.05 else "aviso"
    msg  = (
        "Série <b>estacionária</b> (p < 0.05) — ARIMA pode ser aplicado diretamente."
        if pvalue < 0.05 else
        "Série <b>não estacionária</b> — será necessário diferenciar (d ≥ 1)."
    )
    bloco(msg, tipo)
