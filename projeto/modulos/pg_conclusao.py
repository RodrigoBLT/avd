import streamlit as st
from utils.config import CORES
from utils.helpers import bloco


def render():
    st.title("📖 Conclusão & Storytelling")

    st.markdown("""
    <div class='hero'>
        <h1>A História dos Dados</h1>
        <p>De dados brutos a previsões confiáveis — como o Machine Learning
        transformou uma série histórica de temperaturas em inteligência acionável.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("A Jornada do Projeto")
    etapas = [
        (CORES["primaria"],   "1", "Definição do Problema",    "Identificamos que temperaturas seguem padrões cíclicos previsíveis e que empresas perdem recursos por não antecipá-los."),
        (CORES["secundaria"], "2", "Análise Exploratória",     "Confirmamos sazonalidade anual clara, correlações entre temperatura/umidade e série estacionária via teste ADF."),
        (CORES["ok"],         "3", "Pré-processamento",        "Tratamos dados faltantes com interpolação linear, identificamos outliers climáticos reais e decompusemos a série."),
        (CORES["terciaria"],  "4", "Treinamento dos Modelos",  "Treinamos ARIMA, SARIMA e Prophet — cada modelo com abordagem distinta para tendência e sazonalidade."),
        (CORES["real"],       "5", "Avaliação das Métricas",   "Comparamos MAE, RMSE e MAPE. O Prophet se destacou pela robustez e bandas de confiança probabilísticas."),
        ("#ffffff",           "6", "Forecasting",              "Geramos previsões reais para os próximos meses com intervalos de confiança transparentes e tabela exportável."),
    ]
    for cor, num, titulo, desc in etapas:
        st.markdown(f"""
        <div class='step-row'>
            <div class='step-num' style='background:{cor}18;border:2px solid {cor};color:{cor}'>{num}</div>
            <div class='step-body'>
                <b style='color:{cor}'>{titulo}</b><br>
                <span style='font-size:.88rem'>{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Insights Principais")
    c1, c2 = st.columns(2)
    with c1:
        bloco("<b>Sazonalidade Anual Confirmada</b><br>Ciclos de 12 meses bem definidos — temperatura sobe no verão e cai no inverno com regularidade estatística comprovada pelo teste ADF e decomposição da série.")
        bloco("<b>Prophet como Melhor Modelo</b><br>Menor erro médio e maior robustez a eventos extremos. Indicado para produção com retreinamento mensal.")
    with c2:
        bloco("<b>ARIMA vs SARIMA</b><br>O ARIMA simples \"achata\" previsões longas, perdendo a sazonalidade. O SARIMA corrige isso com os parâmetros P, D, Q — reduzindo o MAPE significativamente.")
        bloco("<b>Incerteza Cresce com o Tempo</b><br>Bandas se alargam quanto mais distante a previsão — comportamento esperado. Previsões de 1–3 meses são confiáveis; acima de 12 requerem cautela.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Comparativo Final de Modelos")
    cols = st.columns(3)
    for col, modelo, pros, contras, cor in [
        (cols[0], "ARIMA",   "Simples, rápido, base sólida",                     "Perde sazonalidade em previsões longas",            CORES["primaria"]),
        (cols[1], "SARIMA",  "Captura ciclos anuais com precisão",                "Requer ajuste fino dos parâmetros sazonais",        CORES["secundaria"]),
        (cols[2], "Prophet", "Robusto, automático, com bandas de incerteza",      "Menos interpretável que ARIMA/SARIMA",              CORES["terciaria"]),
    ]:
        col.markdown(f"""
        <div class='card-metrica' style='text-align:left;padding:16px'>
            <h3 style='color:{cor};font-size:1rem;margin-bottom:10px'>{modelo}</h3>
            <div style='font-size:.82rem;color:#aaa;margin-bottom:6px'>✅ {pros}</div>
            <div style='font-size:.82rem;color:#666'>⚠️ {contras}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Recomendação Final")
    bloco(
        "• Usar o <b>Prophet</b> como modelo principal, retreinado mensalmente com novos dados.<br>"
        "• Considerar previsões de <b>até 6 meses</b> como operacionalmente confiáveis.<br>"
        "• Integrar a previsões a sistemas de planejamento de demanda energética, logística ou agronegócio.<br>"
        "• Monitorar continuamente o <b>desvio entre real e previsto</b> (MAE rolling) para detectar deriva.<br><br>"
        "📊 <b>Resultado esperado:</b> redução de 15–30% nos erros de planejamento operacional relacionados a variações climáticas.",
        "ok",
    )

    st.markdown("""
    <div style='text-align:center;color:#333;font-size:0.78rem;padding:24px 0 8px'>
        Seminário — Tema 6: Séries Temporais e Forecasting &nbsp;·&nbsp;
        Dataset: Weather History (Kaggle) &nbsp;·&nbsp;
        Algoritmos: ARIMA · SARIMA · Prophet
    </div>
    """, unsafe_allow_html=True)
