from fastapi import FastAPI
from metodos import metodos
app = FastAPI(title="API", version="1.0.0")

c = metodos()

@app.get("/index")
def index(): 
    data = c.get_dataset()
    return data

@app.get("/casos_departamentos")
def get_casos_departamentos(): 
    data = c.casos_departamentop()
    return data

@app.get("/armas_medios")
def get_armas_medios(): 
    data = c.armas_medios()
    return data

@app.get("/cantidad")
def get_cantidad(): 
    data = c.cantidad()
    return data

@app.get("/grupo_etario")
def get_grupo_etario(): 
    data = c.grupo_etario()
    return data

@app.get("/genero")
def get_genero(): 
    data = c.genero()
    return data

@app.get("/prediccion")
def get_prediccion(): 
    data = c.prediccion()
    return data

