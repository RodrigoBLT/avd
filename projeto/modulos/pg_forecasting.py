import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from utils.config import CORES
from utils.helpers import nova_fig, estilo_ax, bloco


def render(df_mensal):
    st.title("🔮 Forecasting do Futuro")

    bloco(
        "<b>Entrega final do modelo.</b> O Prophet é treinado com <b>todos os dados históricos</b> "
        "e gera previsões reais com bandas de confiança — elemento essencial de transparência em ML."
    )

    meses_futuro = st.slider("Quantos meses prever no futuro?", 3, 36, 12, 1)

    with st.spinner(f"Gerando previsão para os próximos {meses_futuro} meses…"):
        df_full = pd.DataFrame({
            "ds": df_mensal.index.tz_localize(None),
            "y":  df_mensal.values,
        })
        modelo = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        modelo.fit(df_full)
        futuro_df      = modelo.make_future_dataframe(periods=meses_futuro, freq="MS")
        forecast_final = modelo.predict(futuro_df)

    corte      = df_mensal.index[-1].tz_localize(None)
    futuro_prev = forecast_final[forecast_final["ds"] > corte]

    temp_min = futuro_prev["yhat"].min()
    temp_max = futuro_prev["yhat"].max()
    temp_med = futuro_prev["yhat"].mean()
    mes_frio = futuro_prev.loc[futuro_prev["yhat"].idxmin(), "ds"].strftime("%b/%Y")
    mes_quen = futuro_prev.loc[futuro_prev["yhat"].idxmax(), "ds"].strftime("%b/%Y")
    incert   = (futuro_prev["yhat_upper"] - futuro_prev["yhat_lower"]).mean()

    cols = st.columns(5)
    for col, titulo, valor, desc in [
        (cols[0], "Temp. Média",     f"{temp_med:.1f}°C",  f"Próximos {meses_futuro} meses"),
        (cols[1], "Máxima Prevista", f"{temp_max:.1f}°C",  mes_quen),
        (cols[2], "Mínima Prevista", f"{temp_min:.1f}°C",  mes_frio),
        (cols[3], "Limite Superior", f"{futuro_prev['yhat_upper'].max():.1f}°C", "Pior caso (80%)"),
        (cols[4], "Incerteza Média", f"{incert:.1f}°C",    "Largura da banda"),
    ]:
        col.markdown(
            f"<div class='card-metrica'>"
            f"<h3>{titulo}</h3><div class='valor'>{valor}</div>"
            f"<div class='desc'>{desc}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    hist_recente = df_mensal.iloc[-36:]
    hist_idx = hist_recente.index.tz_localize(None) if hist_recente.index.tz is not None else hist_recente.index
    fig, ax = nova_fig(14, 5)
    ax.plot(hist_idx, hist_recente.values,
            color=CORES["primaria"], linewidth=1.3, alpha=0.8, label="Histórico (3 anos)")
    ax.plot(pd.to_datetime(futuro_prev["ds"]), futuro_prev["yhat"],
            color=CORES["terciaria"], linewidth=2, linestyle="--", label="Previsão Prophet")
    ax.fill_between(
        pd.to_datetime(futuro_prev["ds"]),
        futuro_prev["yhat_lower"], futuro_prev["yhat_upper"],
        color=CORES["terciaria"], alpha=0.18, label="Intervalo de Confiança (80%)",
    )
    ax.axvline(hist_idx[-1], color=CORES["secundaria"],
               linestyle="--", linewidth=1.5, label="Início da Previsão")

    idx_pk = futuro_prev["yhat"].idxmax()
    ax.annotate(
        f"Pico: {temp_max:.1f}°C\n{mes_quen}",
        xy=(pd.to_datetime(futuro_prev.loc[idx_pk, "ds"]), temp_max),
        xytext=(30, 20), textcoords="offset points",
        color="white", fontsize=9,
        arrowprops=dict(arrowstyle="->", color=CORES["secundaria"]),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=CORES["card"], edgecolor=CORES["secundaria"]),
    )

    estilo_ax(ax, f"Forecasting — Próximos {meses_futuro} Meses com Bandas de Confiança")
    ax.set_ylabel("Temperatura (°C)")
    ax.legend(facecolor=CORES["card"], labelcolor="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    bloco(
        "<b>Como ler as bandas:</b> A linha tracejada é a previsão central. "
        "A área sombreada representa o <b>intervalo de 80% de confiança</b>. "
        "Bandas mais largas no futuro distante refletem maior incerteza — honesto e esperado.",
    )

    st.subheader("Previsões Mensais Detalhadas")
    df_tabela = pd.DataFrame({
        "Mês":            pd.to_datetime(futuro_prev["ds"]).dt.strftime("%b/%Y"),
        "Previsão (°C)":  futuro_prev["yhat"].round(2).values,
        "Mínimo (°C)":    futuro_prev["yhat_lower"].round(2).values,
        "Máximo (°C)":    futuro_prev["yhat_upper"].round(2).values,
        "Amplitude (°C)": (futuro_prev["yhat_upper"] - futuro_prev["yhat_lower"]).round(2).values,
    })
    st.dataframe(df_tabela, use_container_width=True)

    csv = df_tabela.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar previsões em CSV",
        data=csv,
        file_name=f"previsao_{meses_futuro}meses.csv",
        mime="text/csv",
    )
