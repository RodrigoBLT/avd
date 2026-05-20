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

    # ── Interpretação de impacto de negócio ─────────────────
    mae  = m["MAE"]
    mape = m["MAPE"]

    if mae <= 1.5:
        nivel   = "🟢 Erro baixo"
        cor_txt = "#3B6D11"
        cor_bg  = "#EAF3DE"
        impacto = (
            f"Com MAE de <b>{mae:.2f}°C</b>, o modelo erra em média menos de 1,5 grau por mês — "
            f"preciso o suficiente para decisões operacionais reais. "
            f"Em planejamento energético, isso equivale a reduzir em até <b>20–30% os erros de "
            f"dimensionamento de capacidade</b>. Em logística, permite antecipar janelas climáticas "
            f"com margem de segurança confiável."
        )
    elif mae <= 3.0:
        nivel   = "🟡 Erro aceitável"
        cor_txt = "#854F0B"
        cor_bg  = "#FAEEDA"
        impacto = (
            f"Com MAE de <b>{mae:.2f}°C</b>, o modelo erra em média até 3 graus por mês — "
            f"adequado para orientação sazonal e planejamento de médio prazo. "
            f"Útil para decisões de estoque sazonais no agronegócio e para alertas de "
            f"picos de demanda energética, mas requer margem de segurança adicional em "
            f"operações críticas."
        )
    else:
        nivel   = "🔴 Erro elevado"
        cor_txt = "#A32D2D"
        cor_bg  = "#FCEBEB"
        impacto = (
            f"Com MAE de <b>{mae:.2f}°C</b>, o modelo apresenta erro acima de 3 graus — "
            f"use como referência de tendência sazonal, não para previsões operacionais precisas. "
            f"Considere ajustar os hiperparâmetros ou aumentar o período de treino."
        )

    st.markdown(
        f"""
        <div style='margin-top:12px; padding:12px 16px; border-radius:8px;
                    background:{cor_bg}; border-left:3px solid {cor_txt};'>
            <div style='font-size:12px; font-weight:600; color:{cor_txt};
                        text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px;'>
                {nivel} — Impacto no mundo real
            </div>
            <div style='font-size:13px; color:{cor_txt}; line-height:1.6;'>
                {impacto}
            </div>
            <div style='font-size:11.5px; color:{cor_txt}; opacity:.75; margin-top:6px;'>
                MAPE de {mape:.1f}% — em previsões de temperatura, valores abaixo de 15% são considerados bons para uso operacional.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bloco(texto, tipo="normal"):
    cls = {"normal": "bloco", "ok": "bloco-ok", "aviso": "bloco-aviso"}.get(tipo, "bloco")
    st.markdown(f"<div class='{cls}'>{texto}</div>", unsafe_allow_html=True)


def secao(titulo, icone=""):
    st.markdown(f"<hr>", unsafe_allow_html=True)
    st.subheader(f"{icone} {titulo}".strip())