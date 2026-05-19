"""
Funções auxiliares compartilhadas entre todas as páginas.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.config import CORES


# ── Métricas ────────────────────────────────────────────────
def calcular_metricas(real, previsto):
    mae  = mean_absolute_error(real, previsto)
    rmse = np.sqrt(mean_squared_error(real, previsto))
    mape = np.mean(np.abs((real - previsto) / real)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ── Matplotlib helpers ───────────────────────────────────────
def estilo_ax(ax, titulo=""):
    ax.set_facecolor(CORES["fundo"])
    ax.set_title(titulo, color="white", fontsize=12, pad=10)
    ax.tick_params(colors="#aaaaaa")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")


def nova_fig(w=14, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CORES["fundo"])
    ax.set_facecolor(CORES["fundo"])
    return fig, ax


# ── Componentes Streamlit ────────────────────────────────────
def cards_metricas(m, label=""):
    if label:
        st.markdown(
            f"<p style='color:#666;font-size:0.82rem;margin-bottom:8px;"
            f"text-transform:uppercase;letter-spacing:.05em'>{label}</p>",
            unsafe_allow_html=True,
        )
    c1, c2, c3 = st.columns(3)
    for col, chave, titulo, desc in [
        (c1, "MAE",  "MAE",  "Erro médio absoluto"),
        (c2, "RMSE", "RMSE", "Erro quadrático médio"),
        (c3, "MAPE", "MAPE", "Erro percentual médio"),
    ]:
        unidade = "%" if chave == "MAPE" else "°C"
        fmt     = f"{m[chave]:.1f}{unidade}" if chave == "MAPE" else f"{m[chave]:.2f}{unidade}"
        col.markdown(
            f"<div class='card-metrica'>"
            f"<h3>{titulo}</h3><div class='valor'>{fmt}</div>"
            f"<div class='desc'>{desc}</div></div>",
            unsafe_allow_html=True,
        )


def bloco(texto, tipo="normal"):
    cls = {"normal": "bloco", "ok": "bloco-ok", "aviso": "bloco-aviso"}.get(tipo, "bloco")
    st.markdown(f"<div class='{cls}'>{texto}</div>", unsafe_allow_html=True)


def secao(titulo, icone=""):
    st.markdown(f"<hr>", unsafe_allow_html=True)
    st.subheader(f"{icone} {titulo}".strip())
