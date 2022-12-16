import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


class metodos:

    def __init__(self):
        self.df = pd.read_csv('database/Violencia_Intrafamiliar_Colombia.csv', nrows=1000, low_memory=False)
        self.df['ARMAS MEDIOS'].fillna(self.df['ARMAS MEDIOS'].mode()[0], inplace = True)
        self.df['GENERO'].fillna(self.df['GENERO'].mode()[0], inplace = True)
        self.df['GRUPO ETARIO'].fillna(self.df['GRUPO ETARIO'].mode()[0], inplace = True)
        self.df['CODIGO DANE'].fillna(self.df['CODIGO DANE'].mode()[0], inplace = True)

    def get_dataset(self):
        return self.df.to_dict(orient="records")

    def casos_departamentop(self):
        casos_departamento = self.df["DEPARTAMENTO"].value_counts().to_frame()
        return casos_departamento.to_dict()['DEPARTAMENTO']

    def armas_medios(self):
        armas_medios = self.df["ARMAS MEDIOS"].value_counts().to_frame()
        return armas_medios.to_dict()['ARMAS MEDIOS']

    def cantidad(self):
        cantidad = self.df["CANTIDAD"].value_counts().to_frame()
        return cantidad.to_dict()['CANTIDAD']

    def grupo_etario(self):
        self.df['GRUPO ETARIO']=self.df['GRUPO ETARIO'].replace({'NO REPORTA': 'ADULTOS'})
        grupo_etario = self.df["GRUPO ETARIO"].value_counts().to_frame()
        return grupo_etario.to_dict()['GRUPO ETARIO']

    def genero(self):
        genero = self.df["GENERO"].value_counts().to_frame()
        return genero.to_dict()['GENERO']

    def prediccion(self):
        trainx=self.df.copy()
        trainx=trainx.drop('ARMAS MEDIOS', axis=1)
        trainy=self.df['ARMAS MEDIOS']
        trainx_encoded = pd.get_dummies(trainx, columns = ['MUNICIPIO', 'DEPARTAMENTO', 'GENERO','GRUPO ETARIO'])
        X_train, X_test, y_train, y_test = train_test_split(trainx_encoded, trainy, test_size=0.2, random_state=42)
        X_train, y_train = make_classification(n_samples=381576, n_features=1059, n_informative=5, n_redundant=5, n_classes=3, random_state=12)
        # define the multinomial logistic regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        # fit the model on the whole dataset
        model.fit(X_train, y_train)
        p = model.score(X_train, y_train)
        return p