import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="wide")

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza predicciones utilizando un modelo de Bagging pre-entrenado.")

# Cargar recursos necesarios de forma global para optimizar
@st.cache_resource
def cargar_recursos():
    one_hot_transformer = joblib.load('one_hot_columns.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    # Usamos bagging_model_optimizado.joblib que es el que existe en el entorno
    model = joblib.load('bagging_model_optimizado.joblib')
    return one_hot_transformer, scaler, model

try:
    one_hot_transformer, scaler, model = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo: {e}")
    st.stop()

# Función para procesar un DataFrame y realizar predicciones
def procesar_y_predecir(df_input):
    df_procesado = df_input.copy()
    
    # 1. Aplicar One-Hot para Felder
    if isinstance(one_hot_transformer, list):
        si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
        for col_name in si_columnas_one_hot:
            valor_esperado = col_name.replace('Felder_', '')
            df_procesado[col_name] = (df_procesado['Felder'].astype(str).str.lower() == valor_esperado.lower()).astype(int)
    else:
        df_encoded = pd.get_dummies(df_procesado[['Felder']])
        df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
        si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]

    # Eliminar variable original Felder
    df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')

    # Asegurar existencia de columnas del modelo
    if isinstance(one_hot_transformer, list):
        for col in si_columnas_one_hot:
            if col not in df_procesado.columns:
                df_procesado[col] = 0

    # 2. Normalizar Examen_admisión
    df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
    df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')

    # Reordenar columnas
    columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
    df_procesado = df_procesado[columnas_ordenadas]

    # 3. Predicción
    predicciones = model.predict(df_procesado)
    return predicciones

# Crear pestañas para separar la predicción individual de la masiva
tab1, tab2 = st.tabs(["📝 Predicción Individual (Manual)", "📊 Predicción Masiva (Subir Excel)"])

with tab1:
    st.header("Datos de Entrada Individual")
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    
    col1, col2 = st.columns(2)
    with col1:
        felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder, key="manual_felder")
    with col2:
        examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01, key="manual_examen")

    if st.button("Realizar Predicción Individual", type="primary"):
        try:
            df_manual = pd.DataFrame([{'Felder': felder_input, 'Examen_admisión': examen_input}])
            prediccion = procesar_y_predecir(df_manual)
            
            st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")
        except Exception as e:
            st.error(f"Ocurrió un error en la predicción manual: {e}")

with tab2:
    st.header("Predicción mediante Archivo Excel")
    st.write("Sube un archivo de Excel (.xlsx) que contenga al menos las columnas: **Felder** y **Examen_admisión**.")
    
    archivo_cargado = st.file_uploader("Selecciona tu archivo Excel", type=["xlsx"])
    
    if archivo_cargado is not None:
        try:
            df_excel = pd.read_excel(archivo_cargado)
            st.write("**Vista previa del archivo cargado:**")
            st.dataframe(df_excel.head(5))
            
            # Validar que existan las columnas necesarias
            columnas_requeridas = ['Felder', 'Examen_admisión']
            columnas_faltantes = [col for col in columnas_requeridas if col not in df_excel.columns]
            
            if columnas_faltantes:
                st.error(f"Faltan las siguientes columnas requeridas en el archivo: {columnas_faltantes}")
            else:
                if st.button("Procesar archivo y realizar predicciones", type="primary"):
                    with st.spinner("Procesando registros..."):
                        # Filtrar datos de entrada para la predicción
                        df_datos = df_excel[columnas_requeridas].dropna().copy()
                        
                        # Realizar predicción
                        resultados_predicciones = procesar_y_predecir(df_datos)
                        
                        # Crear tabla de resultados para el usuario
                        df_resultados = df_datos.copy()
                        df_resultados['Nota_final_Predicha'] = resultados_predicciones
                        
                        st.success("¡Predicciones generadas con éxito!")
                        st.subheader("Resultados Obtenidos")
                        st.dataframe(df_resultados)
                        
                        # Opción de descargar los resultados en un archivo CSV
                        csv = df_resultados.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar resultados en CSV",
                            data=csv,
                            file_name="predicciones_aprobacion.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo Excel: {e}")
