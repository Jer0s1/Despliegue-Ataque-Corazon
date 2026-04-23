# App.py corregido
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# ── Cargar modelo ──────────────────────────────────────────────────────────────
filename = 'modelo-class.pkl'
modelo, abelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# ── UI ────────────────────────────────────────────────────────────────────────
st.title('Predicción de ataque al corazón')

age               = st.slider('Edad', min_value=14, max_value=100, value=50, step=1)
hypertension      = st.selectbox('¿Hipertensión?', ['Yes', 'No'])
heart_disease     = st.selectbox('¿Enfermedad cardíaca previa?', ['Yes', 'No'])
ever_married      = st.selectbox('¿Alguna vez casado/a?', ['Yes', 'No'])
avg_glucose_level = st.number_input('Nivel promedio de glucosa',
                                    min_value=40.0, max_value=600.0,
                                    value=100.0, step=1.0, format="%.1f")
smoking_status    = st.selectbox('Estado de fumador',
                                 ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# ── Predicción ────────────────────────────────────────────────────────────────
if st.button('Predecir'):

    # 1. DataFrame con los datos crudos
    datos = [[age, hypertension, heart_disease, ever_married,
              avg_glucose_level, smoking_status]]
    data = pd.DataFrame(datos,
                        columns=['age', 'hypertension', 'heart_disease',
                                 'ever_married', 'avg_glucose_level', 'smoking_status'])

    # 2. get_dummies  (drop_first=False, igual que en entrenamiento)
    data_prep = pd.get_dummies(data,
                               columns=['hypertension', 'heart_disease',
                                        'ever_married', 'smoking_status'],
                               drop_first=False, dtype=int)

    # 3. Alinear columnas con las del entrenamiento (añade las que falten, elimina extras)
    data_prep = data_prep.reindex(columns=variables, fill_value=0)

    # 4. Escalar SOLO las columnas numéricas (igual que en entrenamiento)
    data_prep[['age', 'avg_glucose_level']] = min_max_scaler.transform(
        data_prep[['age', 'avg_glucose_level']]
    )

    # 5. Predicción
    pred      = modelo.predict(data_prep)[0]
    pred_prob = modelo.predict_proba(data_prep)[0]

    # 6. Mostrar resultado
    if pred == 1:
        st.error(f' Alto riesgo de ataque al corazón  '
                 f'(probabilidad: {pred_prob[1]*100:.1f}%)')
    else:
        st.success(f' Bajo riesgo de ataque al corazón  '
                   f'(probabilidad de riesgo: {pred_prob[1]*100:.1f}%)')

    st.warning('El modelo tiene una precisión del 73%')
    st.dataframe(data.assign(Prediccion=pred))
