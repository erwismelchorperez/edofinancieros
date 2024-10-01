import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error# erros cuadratico medio
from sklearn.metrics import r2_score# coeficiente de determinación
from sklearn.metrics import median_absolute_error# error absoluto medio
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

from openpyxl import Workbook

from deap import base, creator, tools, algorithms
import random

class GeneticAlgorithmRandomForest:
    def __init__(self, population_size, mutation_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                'n_estimators': np.random.randint(10, 200),
                'max_depth': np.random.randint(1, 20),
                'min_samples_split': np.random.randint(2, 20),
                'min_samples_leaf': np.random.randint(1, 20)
            }
            population.append(individual)
        return population

    def fitness(self, individual, X_train, y_train, X_val, y_val):
        model = RandomForestRegressor(
            n_estimators=individual['n_estimators'],
            max_depth=individual['max_depth'],
            min_samples_split=individual['min_samples_split'],
            min_samples_leaf=individual['min_samples_leaf'],
            random_state=42
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        return mean_squared_error(y_val, predictions)
    
    def selection(self, population, fitnesses):
        idx = np.argsort(fitnesses)
        return [population[i] for i in idx[:self.population_size // 2]]

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    def mutation(self, individual):
        for key in individual:
            if random.random() < self.mutation_rate:
                if key == 'n_estimators':
                    individual[key] = np.random.randint(10, 200)
                elif key == 'max_depth':
                    individual[key] = np.random.randint(1, 20)
                elif key == 'min_samples_split':
                    individual[key] = np.random.randint(2, 20)
                elif key == 'min_samples_leaf':
                    individual[key] = np.random.randint(1, 20)
        return individual

    def evolve(self, population, X_train, y_train, X_val, y_val):
        for _ in range(self.generations):
            fitnesses = [self.fitness(ind, X_train, y_train, X_val, y_val) for ind in population]
            selected = self.selection(population, fitnesses)
            children = []
            for i in range(len(selected) // 2):
                parent1, parent2 = selected[2*i], selected[2*i + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                children.append(child)
            population = selected + children
        return population

class GeneticAlgorithmDecisionTree:
    def __init__(self, population_size, mutation_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                'max_depth': np.random.randint(1, 20),
                'min_samples_split': np.random.randint(2, 20),
                'min_samples_leaf': np.random.randint(1, 20)
            }
            population.append(individual)
        return population

    def fitness(self, individual, X_train, y_train, X_val, y_val):
        model = DecisionTreeRegressor(
            max_depth=individual['max_depth'],
            min_samples_split=individual['min_samples_split'],
            min_samples_leaf=individual['min_samples_leaf'],
            random_state=42
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        return mean_squared_error(y_val, predictions)
    
    def selection(self, population, fitnesses):
        idx = np.argsort(fitnesses)
        return [population[i] for i in idx[:self.population_size // 2]]

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    def mutation(self, individual):
        for key in individual:
            if random.random() < self.mutation_rate:
                if key == 'max_depth':
                    individual[key] = np.random.randint(1, 20)
                elif key == 'min_samples_split':
                    individual[key] = np.random.randint(2, 20)
                elif key == 'min_samples_leaf':
                    individual[key] = np.random.randint(1, 20)
        return individual

    def evolve(self, population, X_train, y_train, X_val, y_val):
        for _ in range(self.generations):
            fitnesses = [self.fitness(ind, X_train, y_train, X_val, y_val) for ind in population]
            selected = self.selection(population, fitnesses)
            children = []
            for i in range(len(selected) // 2):
                parent1, parent2 = selected[2*i], selected[2*i + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                children.append(child)
            population = selected + children
        return population

class GeneticAlgorithmMLP:
    def __init__(self, population_size, mutation_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                'hidden_layer_sizes': tuple(np.random.randint(50, 200, size=np.random.randint(1, 4))),
                'activation': random.choice(['relu', 'tanh']),
                'learning_rate_init': 10**np.random.uniform(-4, -2),
                'batch_size': np.random.randint(16, 128)
            }
            population.append(individual)
        return population

    def fitness(self, individual, X_train, y_train, X_val, y_val):
        model = MLPRegressor(hidden_layer_sizes=individual['hidden_layer_sizes'],
                             activation=individual['activation'],
                             learning_rate_init=individual['learning_rate_init'],
                             batch_size=individual['batch_size'],
                             max_iter=200,
                             random_state=42)
        
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_val)
        return mean_squared_error(y_val, predictions)
    
    def selection(self, population, fitnesses):
        idx = np.argsort(fitnesses)
        return [population[i] for i in idx[:self.population_size // 2]]

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    def mutation(self, individual):
        for key in individual:
            if random.random() < self.mutation_rate:
                if key == 'hidden_layer_sizes':
                    individual[key] = tuple(np.random.randint(50, 200, size=np.random.randint(1, 4)))
                elif key == 'activation':
                    individual[key] = random.choice(['relu', 'tanh'])
                elif key == 'learning_rate_init':
                    individual[key] = 10**np.random.uniform(-4, -2)
                elif key == 'batch_size':
                    individual[key] = np.random.randint(16, 128)
        return individual

    def evolve(self, population, X_train, y_train, X_val, y_val):
        for _ in range(self.generations):
            fitnesses = [self.fitness(ind, X_train, y_train, X_val, y_val) for ind in population]
            selected = self.selection(population, fitnesses)
            children = []
            for i in range(len(selected) // 2):
                parent1, parent2 = selected[2*i], selected[2*i + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                children.append(child)
            population = selected + children
        return population

class GeneticAlgorithmLSTM:
    def __init__(self, population_size, mutation_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                'num_units': np.random.randint(50, 200),
                'num_layers': np.random.randint(1, 4),
                'learning_rate': 10**np.random.uniform(-4, -2),
                'batch_size': np.random.randint(16, 128)
            }
            population.append(individual)
        return population

    def fitness(self, individual, X_train, y_train, X_val, y_val):
        model = Sequential()
        for _ in range(individual['num_layers']):
            model.add(LSTM(units=individual['num_units'], activation='relu', return_sequences=True, input_shape=(X_train.shape[0], X_train.shape[1])))
        model.add(LSTM(units=individual['num_units'], activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=individual['learning_rate']), loss='mean_squared_error')
        
        model.fit(X_train, y_train, epochs=10, batch_size=individual['batch_size'], verbose=0)
        
        predictions = model.predict(X_val)
        return mean_squared_error(y_val, predictions)
    
    def selection(self, population, fitnesses):
        idx = np.argsort(fitnesses)
        return [population[i] for i in idx[:self.population_size // 2]]

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
        return child

    def mutation(self, individual):
        for key in individual:
            if random.random() < self.mutation_rate:
                if key == 'num_units':
                    individual[key] = np.random.randint(50, 200)
                elif key == 'num_layers':
                    individual[key] = np.random.randint(1, 4)
                elif key == 'learning_rate':
                    individual[key] = 10**np.random.uniform(-4, -2)
                elif key == 'batch_size':
                    individual[key] = np.random.randint(16, 128)
        return individual

    def evolve(self, population, X_train, y_train, X_val, y_val):
        for _ in range(self.generations):
            fitnesses = [self.fitness(ind, X_train, y_train, X_val, y_val) for ind in population]
            selected = self.selection(population, fitnesses)
            children = []
            for i in range(len(selected) // 2):
                parent1, parent2 = selected[2*i], selected[2*i + 1]
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                children.append(child)
            population = selected + children
        return population

class DecisionTreeModel:
    def __init__(self, max_depth, min_samples_split, min_samples_leaf):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = self.build_model()

    def build_model(self):
        model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        return model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_val, y_val):
        predictions = self.predict(X_val)
        return mean_squared_error(y_val, predictions)

class LSTMModel:
    def __init__(self, input_shape, num_units, num_layers, learning_rate):
        self.input_shape = input_shape
        self.num_units = num_units
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        for _ in range(self.num_layers):
            model.add(LSTM(units=self.num_units, activation='relu', return_sequences=True, input_shape=self.input_shape))
        model.add(LSTM(units=self.num_units, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),  loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_val, y_val):
        predictions = self.predict(X_val)
        return mean_squared_error(y_val, predictions)

class GRUModel:
    def __init__(self, input_shape, num_units, num_layers, learning_rate):
        self.input_shape = input_shape
        self.num_units = num_units
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        for _ in range(self.num_layers):
            model.add(GRU(units=self.num_units, activation='relu', return_sequences=True, input_shape=self.input_shape))
        model.add(GRU(units=self.num_units, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_val, y_val):
        predictions = self.predict(X_val)
        return mean_squared_error(y_val, predictions)

class MLPModel:
    def __init__(self, hidden_layer_sizes, activation, learning_rate_init, batch_size):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.model = self.build_model()

    def build_model(self):
        model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                             activation=self.activation,
                             learning_rate_init=self.learning_rate_init,
                             batch_size=self.batch_size,
                             max_iter=200,
                             random_state=42)
        return model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_val, y_val):
        predictions = self.predict(X_val)
        return mean_squared_error(y_val, predictions)

class Clasificadores:
    def __init__(self, clasificador):
        self.clasificador = clasificador
        self.regresionlineal = LinearRegression()
        self.decisiontree = DecisionTreeRegressor(random_state = 0)
        self.randomforest = RandomForestRegressor(max_depth=2, random_state=0)
        self.mlpregresor = MLPRegressor(random_state=1, max_iter=500)
        
        self.modellstm = Sequential([LSTM(50, return_sequences=True, input_shape=(36, 1)),LSTM(50),Dense(1)])
        self.modellstm.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_squared_error'])

        self.modelgru = Sequential([GRU(50, return_sequences=True, input_shape=(36, 1)),GRU(50),Dense(1)])
        self.modelgru.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_squared_error'])

        self.ypredRL = []
        self.ypredDT = []
        self.ypredRF = []
        self.ypredMLP = []
        self.ypredlstm = []
        self.ypredgru = []
        self.meansquerederror = np.zeros(1)
        self.r2score = np.zeros(1) # es coeficiente
        self.errorabsolutomedio = np.zeros(1)

    
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def EntranientoModelos(self, xtrain):
        xtrain = xtrain.drop(columns=['ejercicio'])
        ytrain = xtrain[xtrain.columns[0]]
        
        """
            RegressionLineal
        """
        if self.clasificador == "RL":
            self.regresionlineal.fit(xtrain, ytrain)    
        else:
            xtrain = self.scaler.fit_transform(np.array(xtrain).reshape(-1, 1))
            ytrain = self.scaler.transform(np.array(ytrain).reshape(-1, 1))
            x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.1, random_state=42)
            if self.clasificador == "DT":
                """
                    DecisionTree
                """
                ga = GeneticAlgorithmDecisionTree(population_size=10, mutation_rate=0.1, generations=5)
                population = ga.initialize_population()
                best_population = ga.evolve(population, x_train, y_train, x_test, y_test)
                best_individual = best_population[0]  # Assuming the first individual is the best
                self.decisiontree = DecisionTreeModel(
                    max_depth=best_individual['max_depth'],
                    min_samples_split=best_individual['min_samples_split'],
                    min_samples_leaf=best_individual['min_samples_leaf']
                )
                self.decisiontree.train(xtrain, ytrain)
            if self.clasificador == "RF":
                """
                    RandomForest
                """
                ga = GeneticAlgorithmRandomForest(population_size=10, mutation_rate=0.1, generations=5)
                population = ga.initialize_population()
                best_population = ga.evolve(population, x_train, y_train, x_test, y_test)
                best_individual = best_population[0]  # Assuming the first individual is the best
                self.randomforest = RandomForestRegressor(
                    n_estimators=best_individual['n_estimators'],
                    max_depth=best_individual['max_depth'],
                    min_samples_split=best_individual['min_samples_split'],
                    min_samples_leaf=best_individual['min_samples_leaf']
                )
                self.randomforest.fit(xtrain, ytrain)
            if self.clasificador == "MLP":
                """
                    MLP
                """
                ga = GeneticAlgorithmMLP(population_size=10, mutation_rate=0.1, generations=5)
                population = ga.initialize_population()
                best_population = ga.evolve(population, x_train, y_train, x_test, y_test)
                # Train the final model with the best hyperparameters
                best_individual = best_population[0]  # Assuming the first individual is the best
                self.mlpregresor = MLPModel(hidden_layer_sizes=best_individual['hidden_layer_sizes'],
                                activation=best_individual['activation'],
                                learning_rate_init=best_individual['learning_rate_init'],
                                batch_size=best_individual['batch_size'])
                self.mlpregresor.train(xtrain, ytrain)
            if self.clasificador == "LSTM":
                """
                    LSTM
                """
                ga = GeneticAlgorithmLSTM(population_size=10, mutation_rate=0.1, generations=5)
                population = ga.initialize_population()
                best_population = ga.evolve(population, x_train, y_train, x_test, y_test)
                best_individual = best_population[0]  # Assuming the first individual is the best
                self.modellstm = LSTMModel(input_shape=(xtrain.shape[0], xtrain.shape[1]),
                                num_units=best_individual['num_units'],
                                num_layers=best_individual['num_layers'],
                                learning_rate=best_individual['learning_rate'])
                self.modellstm.train(xtrain, ytrain, epochs=50, batch_size=best_individual['batch_size'])
            if self.clasificador == "GRU":
                """
                    GRU
                """
                ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1, generations=5)
                population = ga.initialize_population()
                best_population = ga.evolve(population, x_train, y_train, x_test, y_test)
                best_individual = best_population[0]  # Assuming the first individual is the best
                self.modelgru = GRUModel(input_shape=(xtrain.shape[0], xtrain.shape[1]),
                                num_units=best_individual['num_units'],
                                num_layers=best_individual['num_layers'],
                                learning_rate=best_individual['learning_rate'])
                self.modelgru.train(xtrain, ytrain, epochs=50, batch_size=best_individual['batch_size'])
                

    def PrediccionModelos(self, xtest):
        xtest = xtest.drop(columns=['ejercicio'])
        ytest = xtest[xtest.columns[0]]

        #self.ypredRL = self.regresionlineal.predict(xtest)
        #self.ypredDT = self.decisiontree.predict(xtest)
        #self.ypredRF = self.randomforest.predict(xtest)


        xtest = self.scaler.transform(xtest)
        ytestr = self.scaler.transform(np.array(ytest).reshape(-1,1))

        self.ypredDT = self.decisiontree.predict(xtest)
        self.ypredDT = self.scaler.inverse_transform(self.ypredDT.reshape(1,-1))[0]
        #self.ypredMLP = self.mlpregresor.predict(xtest)
        #self.ypredMLP = self.scaler.inverse_transform(self.ypredMLP.reshape(1,-1))[0]
        #self.ypredlstm = self.modellstm.predict(xtest)
        #self.ypredlstm = self.scaler.inverse_transform(self.ypredlstm.reshape(1,-1))[0]
        #self.ypredgru = self.modelgru.predict(xtest)
        #self.ypredgru = self.scaler.inverse_transform(self.ypredgru.reshape(1,-1))[0]

        print("Test: ",ytest)
        #print("RL:   ",self.ypredRL)
        print("DT:   ",self.ypredDT)
        #print("RF:   ",self.ypredRF)
        #print("MLP:  ",self.ypredMLP)
        #print("lstm: ",self.ypredlstm)
        #print("GRU:   ",self.ypredgru)

        self.meansquerederror = [mean_squared_error(self.ypredDT,ytest)]
        self.r2score = [r2_score(self.ypredDT,ytest)]
        self.errorabsolutomedio = [median_absolute_error(self.ypredDT,ytest)]

class EstadosFinancieros:
    def __init__(self, dataset, clasificador):
        self.clasificador = clasificador
        # para construir los encabezados
        #self.modelos = ['RL','DT','RF','MLP','LSTM','GRU']
        self.modelos = ['DT']
        self.meses = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']
        self.errores = ['errorcuadratico','coeficientedeconfianza','errorabsolutomedio']
        # para construir los encabezados
        self.dataset = self.leerdataset(dataset)
        self.columnas = [self.formatearColumns(col) for col in self.dataset.columns]
        self.dataframe_niveles = {}
        self.principales_niveles = []
        self.dataframeFinal = pd.DataFrame()
        self.dataframeFinalErrores = pd.DataFrame()
        self.datos = pd.DataFrame()
        #self.columnasDFF = ['NIVEL', 'CIFRAS AL DIA:', 'ENE22', 'RL_ENE22', 'DT_ENE22', 'RF_ENE22', 'MLP_ENE22', 'FEB22', 'RL_FEB22', 'DT_FEB22', 'RF_FEB22', 'MLP_FEB22', 'MAR22', 'RL_MAR22', 'DT_MAR22', 'RF_MAR22', 'MLP_MAR22', 'ABR22', 'RL_ABR22', 'DT_ABR22', 'RF_ABR22', 'MLP_ABR22', 'MAY22', 'RL_MAY22', 'DT_MAY22', 'RF_MAY22', 'MLP_MAY22', 'JUN22', 'RL_JUN22', 'DT_JUN22', 'RF_JUN22', 'MLP_JUN22', 'JUL22', 'RL_JUL22', 'DT_JUL22', 'RF_JUL22', 'MLP_JUL22', 'AGO22', 'RL_AGO22', 'DT_AGO22', 'RF_AGO22', 'MLP_AGO22', 'SEP22', 'RL_SEP22', 'DT_SEP22', 'RF_SEP22', 'MLP_SEP22', 'OCT22', 'RL_OCT22', 'DT_OCT22', 'RF_OCT22', 'MLP_OCT22', 'NOV22', 'RL_NOV22', 'DT_NOV22', 'RF_NOV22', 'MLP_NOV22', 'DIC22', 'RL_DIC22', 'DT_DIC22', 'RF_DIC22', 'MLP_DIC22']
        self.columnasDFF = self.construirEncabezado()
        #self.columnasErrores = ['NIVEL', 'CIFRAS AL DIA:','RL_errorcuadratico', 'DT_errorcuadratico', 'RF_errorcuadratico', 'MLP_errorcuadratico','RL_coeficientedeconfianza', 'DT_coeficientedeconfianza', 'RF_coeficientedeconfianza', 'MLP_coeficientedeconfianza','RL_errorabsolutomedio', 'DT_errorabsolutomedio', 'RF_errorabsolutomedio', 'MLP_errorabsolutomedio' ]
        self.columnasErrores = self.construirMedidas()
        self.Entranamiento = any
        self.Pruebas = any
        self.DatosGraficar = {}


    def construirEncabezado(self):
        anio = '22'
        columnas = ['NIVEL', 'CIFRAS AL DIA:']
        for mes in self.meses:
            columnas.append(mes+anio)
            for model in self.modelos:
                columnas.append(model+"_"+mes+anio)

        print(len(self.meses))
        return columnas
    
    def construirMedidas(self):
        columnas = ['NIVEL', 'CIFRAS AL DIA:']
        for error in self.errores:
            for model in self.modelos:
                columnas.append(model+'_'+error)
        return columnas

    def formatearColumns(self, col):
        return col.replace('31-','').replace('30-','').replace('29-','').replace('28-','').replace('-','')

    def leerdataset(self, dataset):
        dataset = pd.read_csv(dataset)
        dataset['NIVELv2'] = pd.Series()
        return dataset
    
    def Procesar(self):
        nivel1 = ""
        for i in range(len(self.dataset)):
            if self.dataset.iloc[i]['NIVEL'] == 1.0:
                nivel1 = self.dataset.iloc[i]['CIFRAS AL DIA:'].replace(" ","")
                self.dataset.iloc[i,1] = nivel1
                self.dataset.iloc[i,-1] = "Acumulador"
            if self.dataset.iloc[i]['NIVEL'] == 2.0:
                nivel2 = self.dataset.iloc[i]['CIFRAS AL DIA:'].strip()
                self.dataset.iloc[i,-1] = str(nivel1 + "_" + nivel2)
            if self.dataset.iloc[i]['NIVEL'] == 3.0:
                if self.BusSubcuentaNivel(i, 3):
                    self.dataset.iloc[i,-1] = "Predictor"
                else: 
                    self.dataset.iloc[i,-1] = "Acumulador"
            if self.dataset.iloc[i]['NIVEL'] == 4.0:
                if self.BusSubcuentaNivel(i, 4):
                    self.dataset.iloc[i,-1] = "Predictor"
                else: 
                    self.dataset.iloc[i,-1] = "Acumulador"
            if self.dataset.iloc[i]['NIVEL'] == 5.0:
                if self.BusSubcuentaNivel(i, 5):
                    self.dataset.iloc[i,-1] = "Predictor"
                else: 
                    self.dataset.iloc[i,-1] = "Acumulador"
            if self.dataset.iloc[i]['NIVEL'] == 6.0:
                if self.BusSubcuentaNivel(i, 6):
                    self.dataset.iloc[i,-1] = "Predictor"
                else: 
                    self.dataset.iloc[i,-1] = "Acumulador"

    def BusSubcuentaNivel(self, indice, nivel):
        bandera = False
        if indice == len(self.dataset)-1:
            bandera = True
        else:
            for j in range(indice, len(self.dataset)-1):
                if str(self.dataset.iloc[j]['NIVEL']) == str(nivel+1):
                    break
                if str(self.dataset.iloc[j+1]['NIVEL']) == str(nivel):
                    bandera = True
                    break
        return bandera
    
    def formatearColumns(self,col):
        return col.replace('31-','').replace('30-','').replace('29-','').replace('28-','').replace('-','')
        

    def AnalizarPrediccion(self):
        self.dataframeFinal = pd.DataFrame(data=[], columns=self.columnasDFF)
        elementos = list(np.zeros(shape=len(self.columnasDFF)))
        elementosError = list(np.zeros(shape=len(self.columnasErrores)))
        print("             columnasDFF     ",len(self.columnasDFF))
        print("             columnasDFF     ",self.columnasDFF)
        nivel3 = ""
        nivel4 = ""
        nivel5 = ""
        nivel6 = ""
        #for i in range(5):
        for i in range(len(self.dataset)):
            elementos = list(np.zeros(shape=len(self.columnasDFF)))
            elementosError = list(np.zeros(shape=len(self.columnasErrores)))
            elementos[0] = self.dataset.iloc[i]['NIVEL']
            elementos[1] = self.dataset.iloc[i]['CIFRAS AL DIA:']
            elementosError[0] = self.dataset.iloc[i]['NIVEL']
            elementosError[1] = self.dataset.iloc[i]['CIFRAS AL DIA:']
            if self.dataset.iloc[i]['NIVEL'] == 3.0:
                nivel3 = self.dataset.iloc[i]['CIFRAS AL DIA:'].strip()
                nivel4 = ""
                nivel5 = ""
                nivel6 = ""
            if self.dataset.iloc[i]['NIVEL'] == 4.0:
                nivel4 = self.dataset.iloc[i]['CIFRAS AL DIA:'].strip()
                nivel5 = ""
                nivel6 = ""
            if self.dataset.iloc[i]['NIVEL'] == 5.0:
                nivel5 = self.dataset.iloc[i]['CIFRAS AL DIA:'].strip()
                nivel6 = ""
            if self.dataset.iloc[i]['NIVEL'] == 6.0:
                nivel6 = self.dataset.iloc[i]['CIFRAS AL DIA:'].strip()

            if self.dataset.iloc[i]['NIVELv2'] == 'Predictor':
                
                self.datos = pd.DataFrame(data=[self.dataset.iloc[i]], columns=self.dataset.columns)
                self.datos = self.datos.drop(columns=['NIVEL','NIVELv2'])

                columnas = list(self.datos.columns)
                columnas.pop(0)
                columnas  = [self.formatearColumns(col) for col in columnas]
                #columnas.insert(0,'ejercicio')

                self.datos.set_index('CIFRAS AL DIA:', inplace=True)
                #self.datos.columns = columnas
                self.datos = self.datos.T

                
                newdata = pd.DataFrame(columnas, columns=['ejercicio'])
                newdata.reset_index(inplace=True)
                newdata = newdata.drop(columns=('index'))

                self.datos.reset_index(inplace=True)
                self.datos = self.datos.drop(columns=('index'))
                self.datos = pd.concat([self.datos, newdata], axis=1)
                
                self.Entrenamiento = self.datos[self.datos['ejercicio'].str.contains('19|20|21')]
                self.Pruebas = self.datos[self.datos['ejercicio'].str.contains('22')]

                clasifier = Clasificadores(self.clasificador)
                clasifier.EntranientoModelos(self.Entrenamiento)
                clasifier.PrediccionModelos(self.Pruebas)

                #columnaspruebas = self.Pruebas['ejercicio']
                #print(self.Pruebas[self.Pruebas.columns[0]])
                predict_df = pd.DataFrame(self.Pruebas[self.Pruebas.columns[0]])
                #predict_df['PredictRL'] = clasifier.ypredRL
                predict_df['PredictDT'] = clasifier.ypredDT
                #predict_df['PredictRF'] = clasifier.ypredRF
                #predict_df['PredictMLP'] = clasifier.ypredMLP
                #predict_df['PredictLSTM'] = clasifier.ypredlstm
                #predict_df['PredictGRU'] = clasifier.ypredgru
                predict_df['ejercicio'] = self.Pruebas['ejercicio']
                cadena = ""
                cadena = nivel3 + "_" + nivel4 + "_" + nivel5 + "_" + nivel6
                
                self.DatosGraficar[cadena] = predict_df
                #print("Pruebas              ", len(self.Pruebas))
                #print(self.Pruebas)
                contador = 2
                for j in range(len(self.Pruebas)):
                    elementos[j*2+contador] = self.Pruebas.iloc[j][elementos[1]]
                    #elementos[j*6+contador+1] = clasifier.ypredRL[j]
                    elementos[j*2+contador+1] = clasifier.ypredDT[j]
                    #elementos[j*6+contador+3] = clasifier.ypredRF[j]
                    #elementos[j*6+contador+4] = clasifier.ypredMLP[j]
                    #elementos[j*6+contador+5] = clasifier.ypredlstm[j]
                    #elementos[j*6+contador+6] = clasifier.ypredgru[j]

                    #contador += 1

                newframe = pd.DataFrame(list(elementos))
                newframe = newframe.T
                newframe.columns = self.columnasDFF
                self.dataframeFinal = pd.concat([self.dataframeFinal,newframe])

                """
                    Vamos ha generar el archivo de errores...
                """
                elementosError[2] = clasifier.meansquerederror[0]
                """
                elementosError[3] = clasifier.meansquerederror[1]
                elementosError[4] = clasifier.meansquerederror[2]
                elementosError[5] = clasifier.meansquerederror[3]
                elementosError[6] = clasifier.meansquerederror[4]
                elementosError[7] = clasifier.meansquerederror[5]
                """
                elementosError[3] = clasifier.r2score[0]
                """
                elementosError[9] = clasifier.r2score[1]
                elementosError[10] = clasifier.r2score[2]
                elementosError[11] = clasifier.r2score[3]
                elementosError[12] = clasifier.r2score[4]
                elementosError[13] = clasifier.r2score[5]
                """
                elementosError[4] = clasifier.errorabsolutomedio[0]
                """
                elementosError[15] = clasifier.errorabsolutomedio[1]
                elementosError[16] = clasifier.errorabsolutomedio[2]
                elementosError[17] = clasifier.errorabsolutomedio[3]
                elementosError[18] = clasifier.errorabsolutomedio[4]
                elementosError[19] = clasifier.errorabsolutomedio[5]
                """
                newframeErrores = pd.DataFrame(list(elementosError))
                newframeErrores = newframeErrores.T
                newframeErrores.columns = self.columnasErrores
                self.dataframeFinalErrores = pd.concat([self.dataframeFinalErrores,newframeErrores])
                

            else:
                newframe = pd.DataFrame(list(elementos))
                newframe = newframe.T
                newframe.columns = self.columnasDFF
                self.dataframeFinal = pd.concat([self.dataframeFinal,newframe])

                newframeErrores = pd.DataFrame(list(elementosError))
                newframeErrores = newframeErrores.T
                newframeErrores.columns = self.columnasErrores
                self.dataframeFinalErrores = pd.concat([self.dataframeFinalErrores,newframeErrores])
                
clasificador = "DT"
edo = EstadosFinancieros("./../../../Estados_Financieros.csv",clasificador)


edo.Procesar()
edo.AnalizarPrediccion()
#edo.Pruebas
edo.dataframeFinal.to_csv("EstodosFinancierosPrediccionGA_"+clasificador+".csv")
edo.dataframeFinalErrores.to_csv("EstodosFinancierosPrediccionErroresGA_"+clasificador+".csv")
edo.dataframeFinal.to_excel("EstodosFinancierosPrediccionGA_"+clasificador+".xlsx")
edo.dataframeFinalErrores.to_excel("EstodosFinancierosPrediccionErroresGA_"+clasificador+".xlsx")

#edo.dataframeFinal.head(6)
##### Exportar archivos a dataFrame
for name in edo.DatosGraficar.keys():
    #print(name)
    edo.DatosGraficar[name].to_excel("./DatosGraficarGADT/"+name+"_"+clasificador+".xlsx")

