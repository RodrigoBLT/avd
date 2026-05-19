🌡️ Forecasting Climático — Séries Temporais
Seminário — Tema 6 · Análise de Dados com Machine Learning

Previsão de temperatura com ARIMA, SARIMA e Facebook Prophet.

Estrutura do Projeto
projeto/
├── app.py                  # Ponto de entrada — execute este arquivo
├── weatherHistory.csv      # Dataset (baixar do Kaggle — ver abaixo)
│
├── utils/
│   ├── config.py           # Cores, CSS e layout Plotly compartilhados
│   ├── data.py             # Carregamento e split do dataset
│   └── helpers.py          # Funções auxiliares (métricas, gráficos)
│
└── pages/
    ├── pg_inicio.py        # Página inicial com KPIs e roadmap
    ├── pg_eda.py           # Parte 1 — Definição + EDA
    ├── pg_preprocessamento.py  # Parte 2 — Pipeline de dados
    ├── pg_modelos.py       # Parte 3 — ARIMA, SARIMA, Prophet
    ├── pg_forecasting.py   # Parte 4 — Previsão futura
    ├── pg_conclusao.py     # Conclusão & Storytelling
    └── pg_dashboard.py     # Dashboard interativo com Plotly
Instalação
pip install streamlit pandas numpy matplotlib seaborn \
            statsmodels prophet scikit-learn plotly
Dataset
Acesse: https://www.kaggle.com/datasets/muthuj7/weather-dataset
Baixe weatherHistory.csv
Coloque na pasta raiz do projeto (ao lado de app.py)
Execução
streamlit run app.py
Algoritmos: ARIMA · SARIMA · Facebook Prophet
Visualização: Matplotlib · Plotly (dashboard interativo)
AVD: Princípios de Gestalt aplicados em todos os gráficos
