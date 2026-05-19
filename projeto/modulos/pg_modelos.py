import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from utils.config import CORES, CORES_MODELO
from utils.helpers import nova_fig, estilo_ax, bloco, cards_metricas, calcular_metricas


def render(treino, teste):
    st.title("🤖 Treinamento dos Modelos")

    tab1, tab2, tab3, tab4 = st.tabs(["ARIMA", "SARIMA", "Prophet", "Comparação"])

    with tab1:
        st.header("ARIMA")
        bloco(
            "<b>AutoRegressive Integrated Moving Average</b> — usa valores passados (AR), "
            "diferenciação para estacionaridade (I) e erros passados (MA). "
            "Não captura sazonalidade diretamente."
        )
        c1, c2, c3 = st.columns(3)
        p = c1.slider("p (AR)", 0, 5, 2, key="ap")
        d = c2.slider("d (I)",  0, 2, 1, key="ad")
        q = c3.slider("q (MA)", 0, 5, 2, key="aq")

        with st.spinner("Treinando ARIMA…"):
            mod   = ARIMA(treino, order=(p, d, q)).fit()
            prev  = mod.forecast(steps=len(teste))
            prev  = pd.Series(prev.values, index=teste.index)
            conf  = mod.get_forecast(steps=len(teste)).conf_int()
            bi, bs = conf.iloc[:,0].values, conf.iloc[:,1].values

        cards_metricas(calcular_metricas(teste.values, prev.values), f"ARIMA({p},{d},{q})")
        _plot_previsao(treino, teste, {"ARIMA": (prev, bi, bs)}, f"ARIMA({p},{d},{q}) — Previsão vs Real")

    with tab2:
        st.header("SARIMA")
        bloco(
            "<b>ARIMA Sazonal</b> — adiciona componentes P, D, Q com período m=12, "
            "capturando os ciclos anuais. É o mais indicado para dados climáticos mensais."
        )
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        sp = c1.slider("p", 0, 3, 1, key="sp"); sd = c2.slider("d", 0, 2, 1, key="sd"); sq = c3.slider("q", 0, 3, 1, key="sq")
        sP = c4.slider("P sazonal", 0, 2, 1, key="sP"); sD = c5.slider("D sazonal", 0, 2, 1, key="sD"); sQ = c6.slider("Q sazonal", 0, 2, 1, key="sQ")

        with st.spinner("Treinando SARIMA…"):
            mod   = SARIMAX(treino, order=(sp,sd,sq), seasonal_order=(sP,sD,sQ,12)).fit(disp=False)
            prev  = pd.Series(mod.forecast(steps=len(teste)).values, index=teste.index)
            conf  = mod.get_forecast(steps=len(teste)).conf_int()
            bi, bs = conf.iloc[:,0].values, conf.iloc[:,1].values

        cards_metricas(calcular_metricas(teste.values, prev.values), f"SARIMA({sp},{sd},{sq})×({sP},{sD},{sQ},12)")
        _plot_previsao(treino, teste, {"SARIMA": (prev, bi, bs)}, "SARIMA — Previsão vs Real")

    with tab3:
        st.header("Facebook Prophet")
        bloco(
            "Modelo do Meta baseado em regressão aditiva: <b>y(t) = T(t) + S(t) + H(t) + ε</b>. "
            "Detecta sazonalidades automaticamente, lida com outliers e gera intervalos de confiança probabilísticos."
        )
        c1, c2 = st.columns(2)
        cps = c1.slider("Flexibilidade tendência", 0.01, 0.5, 0.05, 0.01)
        sps = c2.slider("Força sazonalidade",      0.1, 20.0, 10.0, 0.5)

        with st.spinner("Treinando Prophet…"):
            df_pt = pd.DataFrame({"ds": treino.index.tz_localize(None), "y": treino.values})
            mod   = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps,
                            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            mod.fit(df_pt)
            futuro   = mod.make_future_dataframe(periods=len(teste), freq="MS")
            forecast = mod.predict(futuro)
            ft       = forecast.tail(len(teste))
            prev     = pd.Series(ft["yhat"].values, index=teste.index)
            bi, bs   = ft["yhat_lower"].values, ft["yhat_upper"].values

        cards_metricas(calcular_metricas(teste.values, prev.values), "Prophet")
        _plot_previsao(treino, teste, {"Prophet": (prev, bi, bs)}, "Prophet — Previsão vs Real com Bandas de Confiança")

        st.subheader("Componentes Internos do Prophet")
        fig2 = mod.plot_components(forecast)
        fig2.patch.set_facecolor(CORES["fundo"])
        for ax in fig2.axes:
            ax.set_facecolor(CORES["card"])
            ax.tick_params(colors="#aaaaaa")
            ax.title.set_color("white")
        st.pyplot(fig2)
        plt.close()

    with tab4:
        st.header("Comparação Final dos Modelos")

        with st.spinner("Treinando todos os modelos para comparação…"):
            ma = ARIMA(treino, order=(2,1,2)).fit()
            pa = pd.Series(ma.forecast(steps=len(teste)).values, index=teste.index)

            ms = SARIMAX(treino, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            ps = pd.Series(ms.forecast(steps=len(teste)).values, index=teste.index)

            df_pt2 = pd.DataFrame({"ds": treino.index.tz_localize(None), "y": treino.values})
            mp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            mp.fit(df_pt2)
            fc2 = mp.predict(mp.make_future_dataframe(periods=len(teste), freq="MS"))
            pp  = pd.Series(fc2.tail(len(teste))["yhat"].values, index=teste.index)

        met_a = calcular_metricas(teste.values, pa.values)
        met_s = calcular_metricas(teste.values, ps.values)
        met_p = calcular_metricas(teste.values, pp.values)

        df_comp = pd.DataFrame({
            "Modelo":    ["ARIMA(2,1,2)", "SARIMA(1,1,1)×(1,1,1,12)", "Prophet"],
            "MAE (°C)":  [met_a["MAE"],  met_s["MAE"],  met_p["MAE"]],
            "RMSE (°C)": [met_a["RMSE"], met_s["RMSE"], met_p["RMSE"]],
            "MAPE (%)":  [met_a["MAPE"], met_s["MAPE"], met_p["MAPE"]],
        }).round(3)

        st.dataframe(
            df_comp.style.highlight_min(subset=["MAE (°C)", "RMSE (°C)", "MAPE (%)"], color="#1a3a2a"),
            use_container_width=True,
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor(CORES["fundo"])
        modelos_lab = ["ARIMA", "SARIMA", "Prophet"]
        cores_b     = [CORES_MODELO[m] for m in modelos_lab]
        for ax, (label, vals) in zip(axes, [
            ("MAE (°C)",  [met_a["MAE"],  met_s["MAE"],  met_p["MAE"]]),
            ("RMSE (°C)", [met_a["RMSE"], met_s["RMSE"], met_p["RMSE"]]),
            ("MAPE (%)",  [met_a["MAPE"], met_s["MAPE"], met_p["MAPE"]]),
        ]):
            ax.set_facecolor(CORES["fundo"])
            bars = ax.bar(modelos_lab, vals, color=cores_b, edgecolor=CORES["fundo"], width=0.5)
            idx  = int(np.argmin(vals))
            bars[idx].set_edgecolor(CORES["ok"])
            bars[idx].set_linewidth(2.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                        f"{val:.2f}", ha="center", color="white", fontsize=9)
            estilo_ax(ax, label)
            ax.set_ylim(0, max(vals) * 1.3)
        plt.suptitle("Comparação de Métricas — borda verde = vencedor", color="white", fontsize=11, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        _plot_previsao(
            treino, teste,
            {"ARIMA": (pa, None, None), "SARIMA": (ps, None, None), "Prophet": (pp, None, None)},
            "Previsões Sobrepostas no Período de Teste",
        )


def _plot_previsao(treino, teste, modelos_dict, titulo):
    def _naive(idx):
        return idx.tz_localize(None) if idx.tz is not None else idx

    tr_idx = _naive(treino.index[-24:])
    te_idx = _naive(teste.index)

    fig, ax = nova_fig(14, 4.5)
    ax.plot(tr_idx, treino.values[-24:], color="#444455",
            linewidth=0.8, label="Histórico (24m)")
    ax.plot(te_idx, teste.values, color=CORES_MODELO["Real"], linewidth=2, label="Real")

    for nome, (prev, bi, bs) in modelos_dict.items():
        cor = CORES_MODELO.get(nome, CORES["primaria"])
        ax.plot(te_idx, prev.values, color=cor, linestyle="--", linewidth=1.8, label=nome)
        if bi is not None:
            ax.fill_between(te_idx, bi, bs, color=cor, alpha=0.1)

    ax.axvline(tr_idx[-1], color=CORES["secundaria"], linestyle=":", linewidth=1)
    estilo_ax(ax, titulo)
    ax.legend(facecolor=CORES["card"], labelcolor="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
