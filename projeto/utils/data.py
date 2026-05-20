import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Carregando dados…")
def carregar_dados():
    df = pd.read_csv("weatherHistory.csv")
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df = df.sort_values("Formatted Date").reset_index(drop=True)
    df = df.rename(columns={
        "Formatted Date":    "data",
        "Temperature (C)":   "temperatura",
        "Humidity":          "umidade",
        "Wind Speed (km/h)": "vento",
        "Precip Type":       "precipitacao",
    })
    df = df.set_index("data")

    df_diario = df[["temperatura", "umidade", "vento"]].resample("D").mean()
    df_mensal = df["temperatura"].resample("MS").mean()

    df_diario = df_diario.interpolate(method="linear")
    df_mensal = df_mensal.interpolate(method="linear")

    return df_diario, df_mensal


def split_treino_teste(df_mensal, proporcao=0.8):
    corte  = int(len(df_mensal) * proporcao)
    treino = df_mensal.iloc[:corte]
    teste  = df_mensal.iloc[corte:]
    return treino, teste
