import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils.config import CORES
from utils.helpers import nova_fig, estilo_ax, bloco, secao


def render(df_diario, df_mensal):
    st.title("Pré-processamento e Decomposição")

    secao("Tratamento de Nulos")
    nulos    = df_diario.isnull().sum()
    df_proc  = df_diario.interpolate(method="linear")
    nulos2   = df_proc.isnull().sum()

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Antes do tratamento")
        st.dataframe(pd.DataFrame({
            "Variável": nulos.index,
            "Nulos":    nulos.values,
            "% Total":  (nulos.values / len(df_diario) * 100).round(2),
        }), use_container_width=True)
    with c2:
        st.caption("Após interpolação linear")
        st.dataframe(pd.DataFrame({
            "Variável": nulos2.index,
            "Nulos":    nulos2.values,
            "% Total":  (nulos2.values / len(df_proc) * 100).round(2),
        }), use_container_width=True)

    bloco(
        "<b>Interpolação linear</b> respeita a continuidade temporal — "
        "estima valores ausentes proporcionalmente entre o ponto anterior e o posterior.",
        "ok",
    )

    secao("Detecção de Outliers (IQR)")
    Q1  = df_mensal.quantile(0.25)
    Q3  = df_mensal.quantile(0.75)
    IQR = Q3 - Q1
    lim_inf  = Q1 - 1.5 * IQR
    lim_sup  = Q3 + 1.5 * IQR
    outliers = df_mensal[(df_mensal < lim_inf) | (df_mensal > lim_sup)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Limite Inferior", f"{lim_inf:.1f}°C")
    c2.metric("Limite Superior", f"{lim_sup:.1f}°C")
    c3.metric("Outliers encontrados", f"{len(outliers)} meses")

    fig, ax = nova_fig(14, 3)
    ax.plot(df_mensal.index, df_mensal.values, color=CORES["primaria"], linewidth=0.9, alpha=0.8)
    ax.scatter(outliers.index, outliers.values, color=CORES["real"], s=30, zorder=5, label="Outliers")
    ax.axhline(lim_sup, color=CORES["secundaria"], linestyle="--", linewidth=0.8, label="Limite sup.")
    ax.axhline(lim_inf, color=CORES["secundaria"], linestyle="--", linewidth=0.8, label="Limite inf.")
    estilo_ax(ax, "Detecção de Outliers — Temperatura Mensal")
    ax.legend(facecolor=CORES["card"], labelcolor="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    bloco(
        "<b>Decisão: manter os outliers</b> — eventos climáticos extremos (ondas de calor, "
        "frentes frias) representam a realidade. O Prophet lida com eles naturalmente.",
        "aviso",
    )

    secao("Decomposição da Série")
    bloco(
        "A série é decomposta em <b>Tendência</b> (direção geral), "
        "<b>Sazonalidade</b> (ciclo anual de 12 meses) e "
        "<b>Resíduo</b> (ruído aleatório). Resíduo pequeno = modelo aditivo bem ajustado."
    )
    decomp = seasonal_decompose(df_mensal, model="additive", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor(CORES["fundo"])
    for ax, (dados, titulo, cor) in zip(axes, [
        (decomp.observed, "Série Original", CORES["primaria"]),
        (decomp.trend,    "Tendência",      CORES["secundaria"]),
        (decomp.seasonal, "Sazonalidade",   CORES["ok"]),
        (decomp.resid,    "Resíduo",        CORES["real"]),
    ]):
        ax.set_facecolor(CORES["fundo"])
        ax.plot(dados.index, dados.values, color=cor, linewidth=1.1)
        ax.set_ylabel(titulo, color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3a")
    axes[0].set_title("Decomposição Aditiva — Temperatura Mensal", color="white", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    secao("ACF e PACF — Parâmetros do ARIMA")
    df_diff = df_mensal.diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor(CORES["fundo"])
    for ax in axes:
        ax.set_facecolor(CORES["fundo"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3a")
        ax.tick_params(colors="#aaaaaa")
    plot_acf(df_diff,  lags=24, ax=axes[0], color=CORES["primaria"], title="ACF — Parâmetro q")
    plot_pacf(df_diff, lags=24, ax=axes[1], color=CORES["ok"], title="PACF — Parâmetro p", method="ywm")
    for ax in axes:
        ax.title.set_color("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    secao("Divisão Treino / Teste")
    prop = st.slider("Proporção treino:", 0.6, 0.9, 0.8, 0.05)
    c    = int(len(df_mensal) * prop)
    tr, te = df_mensal.iloc[:c], df_mensal.iloc[c:]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total",     f"{len(df_mensal)} meses")
    c2.metric("Treino", f"{len(tr)} meses ({prop*100:.0f}%)")
    c3.metric("Teste",  f"{len(te)} meses ({(1-prop)*100:.0f}%)")

    fig, ax = nova_fig(14, 3.5)
    ax.plot(tr.index, tr.values, color=CORES["primaria"], linewidth=1.2, label="Treino")
    ax.plot(te.index, te.values, color=CORES["real"],     linewidth=1.2, label="Teste")
    ax.axvline(tr.index[-1], color=CORES["secundaria"], linestyle="--", linewidth=1, label="Corte")
    estilo_ax(ax, "Divisão Temporal: Treino vs Teste — ordem cronológica preservada")
    ax.legend(facecolor=CORES["card"], labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    bloco("<b>Regra fundamental:</b> nunca embaralhar dados temporais. "
          "A divisão preserva a cronologia para evitar vazamento de informação futura.", "aviso")
