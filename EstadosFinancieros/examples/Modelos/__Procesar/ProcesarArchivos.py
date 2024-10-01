import pandas as pd
import numpy as np
import os
from openpyxl import Workbook

class ProcesarArchivos:
    def __init__(self):
        self.Errores = pd.DataFrame()
        self.EdoFinanciero = pd.DataFrame()
        self.dataset = pd.DataFrame()
        self.datasetRendimiento = pd.DataFrame()
        self.datasetErrores = pd.DataFrame()
        ###########
        self.errorcuadratico = pd.DataFrame()
        self.coeficientedeconfianza = pd.DataFrame()
        self.errorabsolutomedio = pd.DataFrame()

    def setErrores(self):
        return self.Errores

    def setEdoFinanciero(self):
        return self.EdoFinanciero

    def LeerDataSetRendimiento(self, dataset):
        self.datasetRendimiento = pd.read_csv(dataset)
        self.datasetRendimiento = self.datasetRendimiento.drop(columns=['Unnamed: 0'])

    def LeerDataSetErrores(self, dataset):
        self.datasetErrores = pd.read_csv(dataset)
        self.datasetErrores = self.datasetErrores.drop(columns=['Unnamed: 0'])
        print("Errores shape            ",self.datasetErrores.shape)

    def LeerDataSet(self, dataset):
        self.dataset = pd.read_csv(dataset)
    
    def IniciarPrimerasColumnas(self):
        n = 2
        self.EdoFinanciero = self.dataset.iloc[:, :n]

    def IniciarPrimerasColumnasErrores(self):
        n = 2
        self.errorcuadratico = self.dataset.iloc[:, :n]
        self.coeficientedeconfianza = self.dataset.iloc[:, :n]
        self.errorabsolutomedio = self.dataset.iloc[:, :n]

    def RegresarColumnaErrores(self, cont):
        self.errorcuadratico[cont+'_errorcuadratico'] = self.datasetErrores[cont+'_errorcuadratico']
        self.coeficientedeconfianza[cont+'_coeficientedeconfianza'] = self.datasetErrores[cont+'_coeficientedeconfianza']
        self.errorabsolutomedio[cont+'_errorabsolutomedio'] = self.datasetErrores[cont+'_errorabsolutomedio']
    
    def ImprimirDataset(self):
        print(self.dataset)

    def ImprimirDatasetRendimiento(self):
        print(self.datasetRendimiento)

    def ImprimirDatasetErrores(self):
        print(self.Errores)

    def Encabezados(self):
        return self.dataset.columns

    def RegresarColumna(self, mes, anio):
        self.EdoFinanciero[mes + anio] = self.datasetRendimiento[mes + anio]
        #self.Errores[mes + anio] = self.datasetRendimiento[mes + anio]

    def RegresarColumnaModelo(self, modelo, mes, anio):
        self.EdoFinanciero[modelo + '_' + mes + anio] = self.datasetRendimiento[modelo + '_' + mes + anio]
        #self.Errores[modelo + '_' + mes + anio] = self.datasetRendimiento[modelo + '_' + mes + anio]
    
    def ExpotEdoFinanciero(self):
        self.EdoFinanciero.to_csv("./EstadoFinancieroResultados.csv")
        self.EdoFinanciero.to_excel("./EstadoFinancieroResultados.xlsx")
    
    def ExportErrores(self):
        self.errorcuadratico.to_csv("./ErrorCuadratico.csv")
        self.errorcuadratico.to_excel("./ErrorCuadratico.xlsx")
        self.coeficientedeconfianza.to_csv("./CoeficientedeConfianza.csv")
        self.coeficientedeconfianza.to_excel("./CoeficientedeConfianza.xlsx")
        self.errorabsolutomedio.to_csv("./ErrorAbsolutoMedio.csv")
        self.errorabsolutomedio.to_excel("./ErrorAbsolutoMedio.xlsx")


if __name__ == '__main__':
    current_location = os.path.dirname(os.path.abspath(__file__))
    PathDir = "./../"
    print("Current:   ",current_location)
    contenido = os.listdir(PathDir)
    print("      ",type(contenido))
    contenido.sort(reverse=True)
    print(contenido)
    procesarfile = ProcesarArchivos()
    procesarfile.LeerDataSet("./Estados_Financieros.csv")
    procesarfile.IniciarPrimerasColumnas()
    procesarfile.IniciarPrimerasColumnasErrores()
    anio = "22"
    meses = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']
    for mes in meses:
        bandera = True
        for cont in contenido:    
            if os.path.isdir(os.path.join(PathDir, cont)):
                carpeta = os.path.join(PathDir, cont)
                subcontenido = os.listdir(carpeta)
                if '__Procesar' in carpeta or 'LSTM' in carpeta or 'GRU' in carpeta:
                    print("Carpeta de proceso!!!")
                else:
                    #print("cont         ", cont)
                    #print("cont         ", cont, "  ",carpeta+"/EstodosFinancierosPrediccionGA_"+cont+".csv")
                    procesarfile.LeerDataSetRendimiento(carpeta+"/EstodosFinancierosPrediccionGA_"+cont+".csv")
                    if bandera:
                        procesarfile.RegresarColumna(mes,anio)
                        procesarfile.RegresarColumnaModelo(cont,mes,anio)
                        bandera = False
                    else:
                        procesarfile.RegresarColumnaModelo(cont,mes,anio)


    for cont in contenido:    
        if os.path.isdir(os.path.join(PathDir, cont)):
            carpeta = os.path.join(PathDir, cont)
            subcontenido = os.listdir(carpeta)
            if '__Procesar' in carpeta or 'LSTM' in carpeta or 'GRU' in carpeta:
                print("Carpeta de proceso!!!")
            else:
                print("cont         ", cont, "  ",carpeta+"/EstodosFinancierosPrediccionErroresGA_"+cont+".csv")
                procesarfile.LeerDataSetErrores(carpeta+"/EstodosFinancierosPrediccionErroresGA_"+cont+".csv")
                #procesarfile.IniciarPrimerasColumnasErrores()
                print("Modelo               ", cont)
                procesarfile.RegresarColumnaErrores(cont)
                """
                if bandera:
                    procesarfile.RegresarColumna(mes,anio)
                    procesarfile.RegresarColumnaModelo(cont,mes,anio)
                    bandera = False
                else:
                    procesarfile.RegresarColumnaModelo(cont,mes,anio)
                """
    #procesarfile.LeerDataSetErrores(carpeta+"/EstodosFinancierosPrediccionErroresGA_"+cont+".csv")
            

    #print("\n\n")
    #print(procesarfile.setEdoFinanciero())
    procesarfile.ExpotEdoFinanciero()
    procesarfile.ExportErrores()
    #print(procesarfile.ImprimirDataset())
