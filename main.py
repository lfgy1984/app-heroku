import numpy as np
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

model = load_model("calibrated_lightgbm")


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    label = predictions_df['Label'][0]
    prob = predictions_df['Score'][0] * 100
    return label, prob

labels_dic= {2:'Buen Potencial',
 5:'Retener',
 1:'Alto Potencial',
 3:'Erroneos o Atipicos',
 0:'Algo de Potencial',
 4:'Muy bajo Potencial'}

## funcion para crear caracteristicas sobre el dataframe con variables minimas requeridas
def transform_df(input_df):
    ## porcentaje de ocupacion de los nodos
    input_df['tasa_ocupacion'] = input_df['HP OCUP'] / input_df['HP EFEC']
    ## tasa de crecimiento en rgus de un mes
    input_df['tasa_crecimiento_rgus_home'] = (
                (input_df['RGUS HOME'] - (input_df['RGUS HOME'] - input_df['NETO RGU'])) / (
                    input_df['RGUS HOME'] - input_df['NETO RGU']))
    ## valor promedio facturacion por home pass
    input_df['facturacion_promedio_hp'] = (input_df['FACTURACION'] / input_df['HP OCUP'])
    # Capacidad disponible del nodo
    input_df["capacidad_disponible_nodo"] = input_df["HP EFEC"] - input_df["HP OCUP"]
    # Promedio de servicios por hogar
    input_df["promedio_servicios_hogar"] = input_df["RGUS HOME"] / input_df["HP OCUP"]
    # porcentaje de servicios TO sobre total de servicios
    input_df['TASA_RGUS_TO_HOME'] = input_df['RGUS_TO_HOME'] / input_df['RGUS HOME']
    # porcentaje de servicios BA sobre total de servicios
    input_df['TASA_RGUS_BA_HOME'] = input_df['RGUS_BA_HOME'] / input_df['RGUS HOME']
    # porcentaje de servicios TV sobre total de servicios
    input_df['TASA_RGUS_TV_HOME'] = input_df['RGUS_TV_HOME'] / input_df['RGUS HOME']

    ## df para output
    df = input_df[['tasa_ocupacion', 'tasa_crecimiento_rgus_home',
                   'facturacion_promedio_hp', 'capacidad_disponible_nodo',
                   'promedio_servicios_hogar', 'TASA_RGUS_TO_HOME', 'TASA_RGUS_BA_HOME',
                   'TASA_RGUS_TV_HOME', 'ESTRATO_MODA', 'HP EFEC']]
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return df


def run():
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.write("")

    with col2:
        st.image('tigo.png')


    with col3:
        st.write("")

    st.title('Modelo para Clasificación de nodos según potencial de ventas.')
    st.subheader('Desarrolado por: ')
    st.text('Carlos Parra')
    st.text('Laura López')
    st.text('Lorena Delgado')
    st.text('Luis Garcia')
    st.text('')
    st.text('')

    st.subheader('El modelo da como resultado una de las siguientes etiquetas para cada nodo:')
    st.text('Alto Potencial, Retener, Buen Potencial,')
    st.text('Algo de Potencial, Muy Bajo Potencial y Erróneos o Atípicos.')
    st.text('')
    st.text('')
    st.subheader('También mostrara unas nuevas variables construidas a partir de las que se ingresaron, las cuales son las utilizadas en el modelo y logran resumir datos claves para el negocio.')
    st.subheader('Estas son: ')
    st.text('Facturación promedio por Homepass Efectivo, Facturacion promedio por Homepass Ocupado, Tasa de crecimiento de RGUS, Tasa de ocupación medida en Homepasses, Cantidad de conexiones disponibles para Homepasses, Promedio de RGUS por Homepass Ocupado, Tasa de RGUS telefonía respecto al total, Tasa de RGUS banda ancha respecto al total y Tasa de RGUS televisión respecto al total.')

    add_selectbox = st.sidebar.selectbox("Seleccione modo de uso:", ('Nodo individual', 'Archivo csv para multiples nodos'))
    st.sidebar.info("Puedes ingresar manualmente los datos de un único nodo, o cargar un archivo csv para realizar varias predicciones de manera simultánea.  Los resultados se pueden descargar al final del proceso.")


    if add_selectbox == 'Nodo individual':

        # 'tasa_ocupacion', 'tasa_crecimiento_rgus_home',
        # 'facturacion_promedio_hp', 'capacidad_disponible_nodo',
        # 'promedio_servicios_hogar', 'TASA_RGUS_TO_HOME', 'TASA_RGUS_BA_HOME',
        # 'TASA_RGUS_TV_HOME', 'ESTRATO_MODA','HP EFEC'

        HP_EFEC = st.number_input('Capacidad total en Homepasses del nodo', min_value=0)
        ESTRATO_MODA = st.selectbox('Estrato más representativo en la ubicación del nodo',
                                    ['ESTRATO1', 'ESTRATO2', 'ESTRATO3', 'ESTRATO4', 'ESTRATO5', 'ESTRATO6'])
        RGUS_HOME = st.number_input("RGUS Home", min_value=1)
        FACTURACION = st.number_input("Facturación total del nodo en el mes.", min_value=0)
        HP_OCUP = st.number_input("Homepasses ocupados en el nodo.", min_value=0)
        NETO_RGU = st.number_input("Neto de conexiones para el nodo en el mes.", min_value=0)
        RGUS_TO_HOME = st.number_input("Total de RGUS Telefonía para el nodo.", min_value=0)
        RGUS_BA_HOME = st.number_input("Total de RGUS Banda Ancha para el nodo.", min_value=0)
        RGUS_TV_HOME = st.number_input("Total de RGUS Televisión para el nodo.", min_value=0)




        input_dict = {'FACTURACION': FACTURACION,
                      'HP EFEC': HP_EFEC,
                      'RGUS HOME': RGUS_HOME,
                      'NETO RGU': NETO_RGU,
                      'HP OCUP': HP_OCUP,
                      'RGUS_TO_HOME': RGUS_TO_HOME,
                      'RGUS_BA_HOME': RGUS_BA_HOME,
                      'RGUS_TV_HOME': RGUS_TV_HOME,
                      'ESTRATO_MODA': ESTRATO_MODA}

        input_df = pd.DataFrame([input_dict])
        df = transform_df(input_df)
        if st.button('Predicción'):
            label, prob = predict(model=model, input_df=df)
            label = labels_dic[label]
            st.success('La predicción es: {ou1}, con una certeza del {ou2}%'.format(ou1=str(label), ou2=str(prob)))
            predictions = predict_model(estimator=model, data=df)
            predictions['Label'] = predictions['Label'].map(labels_dic)
            st.write(predictions)
    if add_selectbox == 'Archivo csv para multiples nodos':
        st.subheader('El archivo debe contener de manera obligatoria las siguientes columnas (los nombres deben coincidir exactamente):')
        st.subheader('FACTURACION, HP EFEC')
        st.subheader('RGUS HOME, NETO RGU')
        st.subheader('HP OCUP, RGUS_TO_HOME')
        st.subheader('RGUS_BA_HOME, RGUS_TV_HOME')
        st.subheader('ESTRATO_MODA')
        st.text("")

        file_upload = st.file_uploader('Subir Archivo CSV', type=['csv'])


        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            predictions['Label'] = predictions['Label'].map(labels_dic)
            st.write(predictions)
            file=predictions.to_csv()
            st.download_button(label='Descargar Resultados',data=file,mime='text/csv',file_name='nodos_clasificados.csv')



if __name__ == '__main__':
    run()
