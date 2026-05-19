import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from utils.config import CORES, CORES_MODELO, PLOTLY_LAYOUT
from utils.helpers import calcular_metricas, bloco


def render(df_diario, df_mensal, treino, teste):
    st.title("Dashboard Interativo — Plotly")
    bloco(
        "Visualizações <b>100% interativas</b> com Plotly — zoom, hover, toggle de séries. "
        "Princípios de Gestalt (<i>Continuidade, Similaridade, Figura/Fundo</i>) aplicados em cada gráfico."
    )

    st.subheader("1. Série Histórica — Exploração Temporal")
    st.caption("Gestalt — Continuidade: a linha guia o olho ao longo do tempo.")

    ts = df_mensal.copy()
    ts.index = ts.index.tz_localize(None)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ts.index, y=ts.values, mode="lines", name="Temperatura mensal",
        line=dict(color=CORES_MODELO["ARIMA"], width=1.5),
        fill="tozeroy", fillcolor="rgba(0,200,255,0.06)",
        hovertemplate="%{x|%b/%Y}: <b>%{y:.1f}°C</b><extra></extra>",
    ))
    layout1 = {**PLOTLY_LAYOUT,
        "title": dict(text="Temperatura Média Mensal — Histórico Completo", font=dict(color="white", size=14)),
        "yaxis_title": "°C",
        "xaxis": dict(
            gridcolor=CORES["grid"], zeroline=False, type="date",
            rangeselector=dict(
                bgcolor=CORES["card"], activecolor=CORES["primaria"],
                font=dict(color="white"),
                buttons=[
                    dict(count=1, label="1a",  step="year",  stepmode="backward"),
                    dict(count=3, label="3a",  step="year",  stepmode="backward"),
                    dict(count=5, label="5a",  step="year",  stepmode="backward"),
                    dict(step="all", label="Tudo"),
                ],
            ),
            rangeslider=dict(visible=True, bgcolor=CORES["card"],
                             bordercolor="#2a2d3a", thickness=0.06),
        ),
    }
    fig1.update_layout(**layout1)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2. Heatmap de Sazonalidade — Ano × Mês")
    st.caption("Gestalt — Similaridade: cores iguais agrupam temperaturas equivalentes.")

    df_h = ts.to_frame("temp")
    df_h["ano"] = df_h.index.year
    df_h["mes"] = df_h.index.month
    pivot = df_h.pivot_table(values="temp", index="ano", columns="mes")
    pivot.columns = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

    fig2 = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdBu_r", zmid=float(pivot.values[~np.isnan(pivot.values)].mean()),
        hovertemplate="<b>%{y}</b> — %{x}: <b>%{z:.1f}°C</b><extra></extra>",
        colorbar=dict(
            title=dict(text="°C", font=dict(color=CORES["texto"])),
            tickfont=dict(color=CORES["texto"]),
        ),
    ))
    fig2.update_layout(**{**PLOTLY_LAYOUT,
        "title": dict(text="Heatmap de Temperatura por Ano × Mês", font=dict(color="white", size=14)),
    })
    st.plotly_chart(fig2, use_container_width=True)

    with st.spinner("Treinando modelos para o dashboard…"):
        # ARIMA
        ma2  = ARIMA(treino, order=(2,1,2)).fit()
        pa2  = pd.Series(ma2.forecast(steps=len(teste)).values, index=teste.index)
        ca2  = ma2.get_forecast(steps=len(teste)).conf_int()
        bi_a, bs_a = ca2.iloc[:,0].values, ca2.iloc[:,1].values

        # SARIMA
        ms2  = SARIMAX(treino, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        ps2  = pd.Series(ms2.forecast(steps=len(teste)).values, index=teste.index)
        cs2  = ms2.get_forecast(steps=len(teste)).conf_int()
        bi_s, bs_s = cs2.iloc[:,0].values, cs2.iloc[:,1].values

        # Prophet
        df_pt = pd.DataFrame({"ds": treino.index.tz_localize(None), "y": treino.values})
        mp2   = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        mp2.fit(df_pt)
        fc2   = mp2.predict(mp2.make_future_dataframe(periods=len(teste), freq="MS"))
        ft2   = fc2.tail(len(teste))
        pp2   = pd.Series(ft2["yhat"].values, index=teste.index)
        bi_p, bs_p = ft2["yhat_lower"].values, ft2["yhat_upper"].values

    idx_t = teste.index.tz_localize(None)

    st.subheader("3. Comparação de Modelos — Previsão vs Real")
    st.caption("Gestalt — Similaridade: cada modelo mantém a mesma cor em todos os gráficos.")

    treino_plot = treino.iloc[-24:].copy()
    treino_plot.index = treino_plot.index.tz_localize(None)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=treino_plot.index, y=treino_plot.values, mode="lines",
        name="Histórico (treino)", line=dict(color=CORES_MODELO["Treino"], width=1),
        hovertemplate="%{x|%b/%Y}: %{y:.1f}°C<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=idx_t, y=teste.values, mode="lines", name="Real",
        line=dict(color=CORES_MODELO["Real"], width=2.5),
        hovertemplate="%{x|%b/%Y}: <b>%{y:.1f}°C</b><extra></extra>",
    ))

    for nome, prev, bi, bs in [
        ("ARIMA",   pa2, bi_a, bs_a),
        ("SARIMA",  ps2, bi_s, bs_s),
        ("Prophet", pp2, bi_p, bs_p),
    ]:
        cor = CORES_MODELO[nome]
        r   = int(cor[1:3], 16); g = int(cor[3:5], 16); b = int(cor[5:7], 16)
        fig3.add_trace(go.Scatter(
            x=idx_t, y=prev.values, mode="lines", name=nome,
            line=dict(color=cor, width=1.8, dash="dash"),
            hovertemplate=f"%{{x|%b/%Y}}: %{{y:.1f}}°C<extra>{nome}</extra>",
        ))
        fig3.add_trace(go.Scatter(
            x=list(idx_t) + list(idx_t[::-1]),
            y=list(bs) + list(bi[::-1]),
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip", name=f"IC {nome}",
        ))

    fig3.add_vline(x=treino_plot.index[-1].value // 10**6, line_dash="dot",
                   line_color=CORES["secundaria"], line_width=1,
                   annotation_text="  Início do teste",
                   annotation_font_color=CORES["texto"])
    fig3.update_layout(**{**PLOTLY_LAYOUT,
        "title": dict(text="ARIMA · SARIMA · Prophet vs Real (com intervalos de confiança)",
                      font=dict(color="white", size=14)),
        "yaxis_title": "Temperatura (°C)",
    })
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("4. Comparação de Métricas de Erro")
    st.caption("Gestalt — Figura/Fundo: fundo escuro destaca as barras coloridas em primeiro plano.")

    met_a = calcular_metricas(teste.values, pa2.values)
    met_s = calcular_metricas(teste.values, ps2.values)
    met_p = calcular_metricas(teste.values, pp2.values)

    nomes_m = ["ARIMA", "SARIMA", "Prophet"]
    cores_m = [CORES_MODELO[n] for n in nomes_m]

    fig4 = make_subplots(rows=1, cols=3, subplot_titles=["MAE (°C)", "RMSE (°C)", "MAPE (%)"])
    for col_i, (label, vals) in enumerate([
        ("MAE",  [met_a["MAE"],  met_s["MAE"],  met_p["MAE"]]),
        ("RMSE", [met_a["RMSE"], met_s["RMSE"], met_p["RMSE"]]),
        ("MAPE", [met_a["MAPE"], met_s["MAPE"], met_p["MAPE"]]),
    ], start=1):
        melhor = int(np.argmin(vals))
        for i, (mod, val, cor) in enumerate(zip(nomes_m, vals, cores_m)):
            fig4.add_trace(go.Bar(
                x=[mod], y=[val],
                name=mod if col_i == 1 else None,
                showlegend=(col_i == 1),
                marker=dict(color=cor,
                            line=dict(color=CORES["ok"] if i == melhor else cor,
                                      width=3 if i == melhor else 0)),
                text=[f"{val:.2f}"], textposition="outside",
                textfont=dict(color="white", size=10),
                hovertemplate=f"<b>{mod}</b> {label}: %{{y:.3f}}<extra></extra>",
            ), row=1, col=col_i)

    fig4.update_layout(
        paper_bgcolor=CORES["fundo"], plot_bgcolor=CORES["fundo"],
        font=dict(color=CORES["texto"]),
        legend=dict(bgcolor=CORES["card"], bordercolor="#2a2d3a", borderwidth=1,
                    font=dict(color="white")),
        margin=dict(l=40, r=30, t=70, b=40), barmode="group",
        title=dict(text="Comparação de Métricas — borda verde = melhor modelo",
                   font=dict(color="white", size=14)),
    )
    for i in range(1, 4):
        fig4.update_xaxes(showgrid=False, row=1, col=i)
        fig4.update_yaxes(gridcolor=CORES["grid"], row=1, col=i)
    for ann in fig4.layout.annotations:
        ann.font.color = CORES["texto"]
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("5. Forecasting Futuro — Série Completa")
    st.caption("Gestalt — Continuidade: linha histórica flui sem corte para a previsão futura.")

    n_meses = st.slider("Meses a prever:", 3, 36, 12, 1, key="plotly_slider")

    with st.spinner("Gerando forecast…"):
        df_full = pd.DataFrame({"ds": df_mensal.index.tz_localize(None), "y": df_mensal.values})
        mf      = Prophet(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                          yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        mf.fit(df_full)
        fc_f    = mf.predict(mf.make_future_dataframe(periods=n_meses, freq="MS"))

    corte2      = df_mensal.index[-1].tz_localize(None)
    fut_plot    = fc_f[fc_f["ds"] > corte2]
    ts2         = df_mensal.copy(); ts2.index = ts2.index.tz_localize(None)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=ts2.index, y=ts2.values, mode="lines", name="Histórico real",
        line=dict(color=CORES_MODELO["ARIMA"], width=1.3),
        hovertemplate="%{x|%b/%Y}: <b>%{y:.1f}°C</b><extra></extra>",
    ))
    r, g, b = int(CORES["terciaria"][1:3],16), int(CORES["terciaria"][3:5],16), int(CORES["terciaria"][5:7],16)
    fig5.add_trace(go.Scatter(
        x=list(pd.to_datetime(fut_plot["ds"])) + list(pd.to_datetime(fut_plot["ds"])[::-1]),
        y=list(fut_plot["yhat_upper"]) + list(fut_plot["yhat_lower"][::-1]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo de confiança (80%)", hoverinfo="skip",
    ))
    fig5.add_trace(go.Scatter(
        x=pd.to_datetime(fut_plot["ds"]), y=fut_plot["yhat"],
        mode="lines+markers", name="Previsão (Prophet)",
        line=dict(color=CORES["terciaria"], width=2.5, dash="dash"),
        marker=dict(size=5, color=CORES["terciaria"]),
        hovertemplate="%{x|%b/%Y}: <b>%{y:.1f}°C</b><extra></extra>",
    ))

    idx_pk = fut_plot["yhat"].idxmax()
    fig5.add_vline(x=corte2.value // 10**6, line_dash="dot", line_color=CORES["secundaria"],
                   line_width=1.5, annotation_text="  Início da previsão",
                   annotation_font_color=CORES["texto"])
    fig5.add_annotation(
        x=pd.to_datetime(fut_plot.loc[idx_pk, "ds"]), y=fut_plot.loc[idx_pk, "yhat"],
        text=f"Pico: {fut_plot.loc[idx_pk,'yhat']:.1f}°C",
        showarrow=True, arrowhead=2, arrowcolor=CORES["secundaria"], ax=0, ay=-40,
        font=dict(color="white", size=9),
        bgcolor=CORES["card"], bordercolor=CORES["secundaria"], borderwidth=1,
    )
    fig5.update_layout(**{**PLOTLY_LAYOUT,
        "title": dict(text=f"Forecasting — Próximos {n_meses} Meses com Intervalos de Confiança (80%)",
                      font=dict(color="white", size=14)),
        "yaxis_title": "Temperatura (°C)",
        "xaxis": dict(
            gridcolor=CORES["grid"], zeroline=False, type="date",
            rangeslider=dict(visible=True, bgcolor=CORES["card"],
                             bordercolor="#2a2d3a", thickness=0.05),
        ),
    })
    st.plotly_chart(fig5, use_container_width=True)

    bloco(
        "<b>Dashboard Plotly completo.</b> Todos os gráficos são interativos — passe o cursor para ver valores, "
        "clique nas legendas para ativar/desativar séries, use o range slider para navegar no tempo.",
        "ok",
    )
