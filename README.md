# Forecasting Climático — Séries Temporais
Previsão de temperatura com ARIMA, SARIMA e Facebook Prophet, com dashboard interativo e princípios de Gestalt aplicados à visualização.

## Estrutura do Projeto

```
projeto/
├── app.py                       # Ponto de entrada — execute este arquivo
├── weatherHistory.csv           # Dataset (baixar do Kaggle — ver abaixo)
│
├── utils/
│   ├── config.py                # Cores, CSS e layout Plotly compartilhados
│   ├── data.py                  # Carregamento e split do dataset
│   └── helpers.py               # Métricas, gráficos e interpretação de negócio
│
└── modulos/
    ├── pg_inicio.py             # Página inicial com KPIs e roadmap
    ├── pg_eda.py                # Definição do problema + EDA
    ├── pg_preprocessamento.py  # Pipeline de dados e decomposição
    ├── pg_modelos.py            # ARIMA, SARIMA e Prophet com AIC
    ├── pg_forecasting.py        # Previsão futura com bandas de confiança
    ├── pg_conclusao.py          # Conclusão, storytelling e referências
    └── pg_dashboard.py          # Dashboard interativo com Plotly
```

---

## Como rodar

### 1. Clone o repositório
```bash
git clone <url-do-repo>
cd avd
```

### 2. Crie e ative o ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Instale as dependências
```bash
pip install streamlit pandas numpy matplotlib seaborn statsmodels prophet scikit-learn plotly
```

### 4. Baixe o dataset
1. Acesse: https://www.kaggle.com/datasets/muthuj7/weather-dataset
2. Baixe o arquivo `weatherHistory.csv`
3. Coloque dentro da pasta `projeto/`, ao lado do `app.py`

### 5. Execute
```bash
cd projeto
streamlit run app.py
```

O app abre automaticamente em `http://localhost:8501`.

---

## Funcionalidades

| Página | O que mostra |
|---|---|
| Início | KPIs do dataset e roadmap do projeto |
| EDA | Estatísticas descritivas, sazonalidade, teste ADF |
| Pré-processamento | Tratamento de nulos, outliers, decomposição, ACF/PACF |
| Modelos | ARIMA, SARIMA e Prophet com métricas e AIC |
| Forecasting | Previsão futura com bandas de confiança (80%) |
| Conclusão | Storytelling, recomendações e referências bibliográficas |
| Dashboard | 5 gráficos Plotly interativos com range slider e hover |

---

## Modelos implementados

- **ARIMA** — AutoRegressive Integrated Moving Average, parâmetros via ACF/PACF e AIC
- **SARIMA** — ARIMA com componente sazonal (período 12 meses)
- **Prophet** — modelo do Meta, robusto a outliers, com decomposição automática

Métricas de avaliação: **MAE**, **RMSE** e **MAPE** — com interpretação de impacto no mundo real.

---

## Referências

- Hyndman, R.J. & Athanasopoulos, G. — *Forecasting: Principles and Practice* (otexts.com/fpp3)
- Géron, A. — *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow* (O'Reilly)
- Facebook Prophet Documentation — facebook.github.io/prophet
- Dataset: Weather History — kaggle.com/datasets/muthuj7/weather-dataset
- Plotly — Interactive Time Series — plotly.com/python/time-series
