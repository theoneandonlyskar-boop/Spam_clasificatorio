"""
Clasificador de Correos Spam - Trabajo Final Módulo Machine Learning (EAM)
Integrantes: Cristian David Quintero Ordoñez y Jacobo Ortiz Correa

Aplica la misma pipeline del notebook a datos nuevos:
  Paso 0: eliminar la variable objetivo
  Paso 1: eliminar variables irrelevantes / redundantes
  Paso 2: One-Hot Encoding de EmailDomain (columnas guardadas)
  Paso 3: imputación, recorte de outliers y MinMaxScaler (guardados)
  Paso 4: predicción con el Random Forest optimizado (guardado)
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clasificador de Spam", page_icon="📧", layout="centered")

# ------------------------------------------------------------------
# Cargar los artefactos guardados con joblib
# ------------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    return {
        "modelo": joblib.load("random_forest_optimizado.joblib"),
        "scaler": joblib.load("min_max_scaler.joblib"),
        "columnas_modelo": joblib.load("one_hot_columns.joblib"),
        "columnas_escaladas": joblib.load("columnas_escaladas.joblib"),
        "medianas": joblib.load("medianas_imputacion.joblib"),
        "limites": joblib.load("limites_outliers.joblib"),
    }

try:
    art = cargar_artefactos()
except FileNotFoundError as e:
    st.error(f"No se encontró un archivo del modelo: {e.filename}. "
             "Coloca los 6 archivos .joblib en la misma carpeta que streamlit_app.py.")
    st.stop()

COLS_ELIMINAR = ["RecordID", "FullName", "Phone", "ZodiacSign",
                 "FavoriteColor", "Hobby", "capital_run_length_total"]

# Dominios conocidos por el modelo (sacados de las columnas del One-Hot).
# domain1.com no tiene columna porque drop_first=True la eliminó (es la categoría base).
DOMINIOS = sorted(
    ["domain1.com"] + [c.replace("EmailDomain_", "") for c in art["columnas_modelo"]
                       if c.startswith("EmailDomain_")],
    key=lambda d: int("".join(ch for ch in d if ch.isdigit()) or 0),
)


# ------------------------------------------------------------------
# Pipeline de preprocesamiento (igual al del notebook)
# ------------------------------------------------------------------
def preprocesar(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Paso 0: eliminar la variable objetivo si viene
    df = df.drop(columns=["spam"], errors="ignore")

    # Paso 1: eliminar variables irrelevantes y redundante
    df = df.drop(columns=COLS_ELIMINAR, errors="ignore")

    # Paso 2: One-Hot Encoding con las columnas del entrenamiento
    if "EmailDomain" in df.columns:
        df = pd.get_dummies(df, columns=["EmailDomain"])
    df = df.reindex(columns=art["columnas_modelo"], fill_value=0)
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    # Paso 3: imputar, recortar outliers y escalar
    num = art["columnas_escaladas"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").fillna(art["medianas"])
    lim = art["limites"]
    out_cols = list(lim["lim_inf"].index)
    df[out_cols] = df[out_cols].clip(lower=lim["lim_inf"], upper=lim["lim_sup"], axis=1)
    df[num] = art["scaler"].transform(df[num])
    return df


def predecir(df_raw: pd.DataFrame) -> pd.DataFrame:
    X = preprocesar(df_raw)
    prob = art["modelo"].predict_proba(X)[:, 1]
    pred = art["modelo"].predict(X)
    return pd.DataFrame({
        "Prob_spam": prob.round(3),
        "Predicción": np.where(pred == 1, "SPAM", "No spam"),
    })


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
st.title("📧 Clasificador de Correos Spam")
st.caption("Modelo: Random Forest optimizado con GridSearchCV · Metodología CRISP-DM · "
           "EAM Institución Universitaria")

tab_manual, tab_archivo, tab_info = st.tabs(["Un correo", "Archivo CSV", "Sobre el modelo"])

# ---------------- Pestaña 1: un correo ----------------
with tab_manual:
    st.subheader("Características del correo")
    st.write("Frecuencias relativas (%) de palabras y caracteres dentro del correo.")

    c1, c2 = st.columns(2)
    with c1:
        free = st.number_input("Frecuencia de 'free'", 0.0, 20.0, 0.0, 0.1)
        money = st.number_input("Frecuencia de 'money'", 0.0, 20.0, 0.0, 0.1)
        credit = st.number_input("Frecuencia de 'credit'", 0.0, 20.0, 0.0, 0.1)
        you = st.number_input("Frecuencia de 'you'", 0.0, 10.0, 0.0, 0.1)
        email = st.number_input("Frecuencia de 'email'", 0.0, 10.0, 0.0, 0.1)
    with c2:
        offer = st.number_input("Frecuencia de 'offer'", 0.0, 10.0, 0.0, 0.1)
        remove = st.number_input("Frecuencia de 'remove'", 0.0, 10.0, 0.0, 0.1)
        business = st.number_input("Frecuencia de 'business'", 0.0, 10.0, 0.0, 0.1)
        excl = st.number_input("Frecuencia de '!'", 0.0, 10.0, 0.0, 0.05)
        dollar = st.number_input("Frecuencia de '$'", 0.0, 10.0, 0.0, 0.05)

    st.write("Uso de mayúsculas")
    c3, c4 = st.columns(2)
    with c3:
        cap_avg = st.number_input("Longitud promedio de secuencias en mayúscula", 1.0, 1000.0, 2.0, 0.5)
    with c4:
        cap_long = st.number_input("Secuencia más larga en mayúscula", 0, 100000, 10, 1)

    dominio = st.selectbox("Dominio del remitente", DOMINIOS + ["Otro / desconocido"])

    if st.button("Clasificar correo", type="primary"):
        correo = pd.DataFrame([{
            "word_freq_free": free, "word_freq_money": money, "word_freq_credit": credit,
            "word_freq_you": you, "word_freq_email": email, "word_freq_offer": offer,
            "word_freq_remove": remove, "word_freq_business": business,
            "char_freq_excl": excl, "char_freq_dollar": dollar,
            "capital_run_length_average": cap_avg, "capital_run_length_longest": cap_long,
            "EmailDomain": dominio if dominio != "Otro / desconocido" else "desconocido",
        }])
        res = predecir(correo).iloc[0]
        prob = float(res["Prob_spam"])

        if res["Predicción"] == "SPAM":
            st.error(f"🚫 **SPAM** — probabilidad de spam: {prob:.1%}")
        else:
            st.success(f"✅ **No spam** — probabilidad de spam: {prob:.1%}")
        st.progress(prob)

# ---------------- Pestaña 2: archivo CSV ----------------
with tab_archivo:
    st.subheader("Clasificar varios correos")
    st.write("Sube un CSV con el mismo formato de `Email_Spam.csv` o `DatosPrueba.csv`. "
             "Se aceptan nulos, outliers y dominios nuevos: la pipeline los procesa igual que en el notebook.")

    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is not None:
        datos = pd.read_csv(archivo)
        st.write(f"**{len(datos)} correos cargados.** Vista previa:")
        st.dataframe(datos.head(10), use_container_width=True)

        faltantes = [c for c in art["columnas_escaladas"] if c not in datos.columns]
        if faltantes:
            st.warning(f"Faltan columnas: {', '.join(faltantes)}. Se imputarán con la mediana.")

        resultado = predecir(datos)
        if "RecordID" in datos.columns:
            resultado.insert(0, "RecordID", datos["RecordID"].values)

        n_spam = int((resultado["Predicción"] == "SPAM").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Correos", len(resultado))
        m2.metric("Spam", n_spam)
        m3.metric("No spam", len(resultado) - n_spam)

        st.dataframe(resultado, use_container_width=True)
        st.download_button("Descargar predicciones (CSV)",
                           resultado.to_csv(index=False).encode("utf-8"),
                           "predicciones_spam.csv", "text/csv")

# ---------------- Pestaña 3: información ----------------
with tab_info:
    st.subheader("Sobre el modelo")
    st.markdown("""
**Problema:** clasificación binaria de correos (1 = spam, 0 = no spam).
**Dataset:** `Email_Spam.csv` — 2.000 correos, 35 % spam.

**Pipeline de preprocesamiento**
1. Eliminar la variable objetivo y las variables irrelevantes (RecordID, FullName, Phone, ZodiacSign, FavoriteColor, Hobby).
2. Eliminar `capital_run_length_total` por redundancia (r = 0,85 con el promedio).
3. One-Hot Encoding de `EmailDomain`.
4. Imputación de nulos con la mediana.
5. Recorte de outliers (IQR) en las variables de mayúsculas.
6. Normalización con MinMaxScaler.

**Modelo final:** Random Forest optimizado con GridSearchCV
(100 árboles, `min_samples_split=5`).

| Métrica (CV 5 folds) | Valor |
|---|---|
| Accuracy | 0,885 |
| Precision | 0,874 |
| Recall | 0,783 |
| F1-score | 0,826 |
| ROC-AUC | 0,916 |

**Variables más importantes:** frecuencia de *credit*, *free*, *$* y *money*.
""")
