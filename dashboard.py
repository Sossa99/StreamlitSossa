
from pandas import json_normalize
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

st.header("DASHBOARD")
st.sidebar.header("INFORMACION DE LOS CASOS DE VIOLENCIA INTRAFAMILIAR")
st.sidebar.markdown("---")

servidor = 'http://44.212.209.121:8000'

@st.cache
def index():
    response = requests.get(servidor + '/index')
    info = json.loads(response.text)    
    datos = json_normalize(info)
    return datos

def get_casos_departamentos():
    response = requests.get(servidor + '/casos_departamentos')
    info = json.loads(response.text)    
    datos = json_normalize(info)
    return datos

def get_armas_medios():
    response = requests.get(servidor + '/armas_medios')
    info = json.loads(response.text)    
    datos = json_normalize(info)
    return datos

def get_cantidad():
    response = requests.get(servidor + '/cantidad')
    info = json.loads(response.text)    
    datos = json_normalize(info)
    return datos

def get_grupo_etario():
    response = requests.get(servidor + '/grupo_etario')
    info = json.loads(response.text)    
    datos = json_normalize(info)
    return datos

def get_genero():
    response = requests.get(servidor + '/genero')
    info = json.loads(response.text)    
    datos = json_normalize(info)
    return datos

def get_prediccion():
    response = requests.get(servidor + '/prediccion')
    datos = response.text
    return datos


datos = index()

radio_button = st.sidebar.radio(
    label="Mostrar dataset", options=["SI", "NO"]
)

if radio_button == "SI":
    st.write('Dataset a analizar')
    st.dataframe(datos)



opciones_select = ['Tabla', 'Grafica', 'Todo']
select_cd = st.sidebar.selectbox(
    label="Casos por departamento", options=opciones_select
)

select_as = st.sidebar.selectbox(
    label="Armas usadas", options=opciones_select
)

select_ge = st.sidebar.selectbox(
    label="Grupos estarios", options=opciones_select
)

select_cantidad = st.sidebar.selectbox(
    label="Cantidad de participantes por casos", options=opciones_select
)

select_genero = st.sidebar.selectbox(
    label="Generos por casos", options=opciones_select
)


#---------- CASOS POR DEPARTAMENTO----------------------------------
st.write('Cantidad de casos por departamento')
casos_departamentos = get_casos_departamentos() 
#-----------------tratamiento para mostratrar la grafica-------------------
aux_casos_departamentos = casos_departamentos.transpose()
aux_casos_departamentos = aux_casos_departamentos.reset_index()
aux_casos_departamentos.rename(columns = {'index':'DEPARTAMENTOS',  0:'N_CASOS'}, inplace = True) 
fig_aux_casos_departamentos = px.bar(aux_casos_departamentos, x="DEPARTAMENTOS", y="N_CASOS", text_auto='.2s', color='DEPARTAMENTOS')
fig_aux_casos_departamentos.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
#----------------- fin tratamiento para mostratrar la grafica-------------------

if select_cd == opciones_select[0]:
    st.write(casos_departamentos)
elif select_cd == opciones_select[1]:
    st.plotly_chart(fig_aux_casos_departamentos, use_container_width=True)
elif select_cd == opciones_select[2]:
    st.write(casos_departamentos)
    st.plotly_chart(fig_aux_casos_departamentos, use_container_width=True)
#---------- CASOS POR DEPARTAMENTO----------------------------------



#---------- ARMAS USADAS----------------------------------
st.write('Armas usadas en los casos de maltrato')
armas_medios = get_armas_medios() 
#-----------------tratamiento para mostratrar la grafica-------------------
aux_armas_medios = armas_medios.transpose()
aux_armas_medios = aux_armas_medios.reset_index()
aux_armas_medios.rename(columns = {'index':'ARMA',  0:'CANTIDAD'}, inplace = True) 
fig_aux_armas_medios = px.bar(aux_armas_medios, x="ARMA", y="CANTIDAD", text_auto='.2s', color='ARMA')
fig_aux_armas_medios.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
#----------------- fin tratamiento para mostratrar la grafica-------------------

if select_as == opciones_select[0]:
    st.write(armas_medios)
elif select_as == opciones_select[1]:
    st.plotly_chart(fig_aux_armas_medios, use_container_width=True)
elif select_as == opciones_select[2]:
    st.write(armas_medios)
    st.plotly_chart(fig_aux_armas_medios, use_container_width=True)
#---------- CASOS ARMAS USADAS----------------------------------



#---------- GRUPOS ETARIOS----------------------------------
st.write('Grupos etarios y cantidad de casos involucados')
grupo_etario = get_grupo_etario() 

#-----------------tratamiento para mostratrar la grafica-------------------
aux_grupo_etario = grupo_etario.transpose()
aux_grupo_etario = aux_grupo_etario.reset_index()
aux_grupo_etario.rename(columns = {'index':'GRUPO',  0:'CANTIDAD'}, inplace = True) 
fig_aux_grupo_etario = px.pie(aux_grupo_etario, values="CANTIDAD", names="GRUPO", color_discrete_sequence=px.colors.sequential.RdBu) 
#----------------- fin tratamiento para mostratrar la grafica-------------------

if select_ge == opciones_select[0]:
    st.write(grupo_etario)
elif select_ge == opciones_select[1]:
    st.plotly_chart(fig_aux_grupo_etario, use_container_width=True)
elif select_ge == opciones_select[2]:
    st.write(grupo_etario)
    st.plotly_chart(fig_aux_grupo_etario, use_container_width=True)
#---------- FIN GRUPOS ETARIOS----------------------------------



#---------- CANTIDAD ----------------------------------
st.write('Cantidad de participantes por casos')
cantidad = get_cantidad() 
#-----------------tratamiento para mostratrar la grafica-------------------
aux_cantidad = cantidad.transpose()
aux_cantidad = aux_cantidad.reset_index()
aux_cantidad.rename(columns={'index':'N_PERSONAS',  0:'CANTIDAD'}, inplace = True) 
fig_aux_cantidad = px.line(x=aux_cantidad['N_PERSONAS'], y=aux_cantidad['CANTIDAD'], labels={'x':'N_PERSONAS', 'y':'CANTIDAD'})
#----------------- fin tratamiento para mostratrar la grafica-------------------

if select_cantidad == opciones_select[0]:
    st.write(cantidad)  
elif select_cantidad == opciones_select[1]:
    st.plotly_chart(fig_aux_cantidad, use_container_width=True)
elif select_cantidad == opciones_select[2]:
    st.write(cantidad)  
    st.plotly_chart(fig_aux_cantidad, use_container_width=True)
#---------- FIN CANTIDAD----------------------------------


#---------- GENEROS----------------------------------
st.write('Casos por genero')
genero = get_genero()
#-----------------tratamiento para mostratrar la grafica-------------------
aux_genero = genero.transpose()
aux_genero = aux_genero.reset_index()
aux_genero.rename(columns = {'index':'GENERO',  0:'CANTIDAD'}, inplace = True) 
fig_aux_genero = px.pie(aux_genero, values="CANTIDAD", names="GENERO", color_discrete_sequence=px.colors.sequential.RdBu, hole=.3) 
#----------------- fin tratamiento para mostratrar la grafica-------------------

if select_genero == opciones_select[0]:
    st.write(genero)
elif select_genero == opciones_select[1]:
    st.plotly_chart(fig_aux_genero, use_container_width=True)
elif select_genero == opciones_select[2]:
    st.write(genero)
    st.plotly_chart(fig_aux_genero, use_container_width=True)
#---------- FIN GENEROS----------------------------------

if st.sidebar.button('Realizar regresion logistica'):
    porcentaje = get_prediccion()
    st.write("Porcentaje de prediocción")
    st.write(porcentaje)
else:
     st.write("Porcentaje de prediocción (debe seleccionar el boton)")
