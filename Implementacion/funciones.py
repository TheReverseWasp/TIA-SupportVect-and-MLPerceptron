#Hecho por Ricardo Manuel Lazo Vásquez - Universidad Catolica San Pablo - TheReverseWasp (GitHub)
#Esta implementacion busca implementar un Perceptron Multicapa ademas de una Support Vector Machine

import numpy as np
import pandas as pd
import sklear as skl

#constante para compilacion de logaritmos
eps = 1e-10

#Funcion para leer iris dataset
#Solo funciona con iris.csv asi que si se quiere atacar otro dataset modificar esta o crear otra función
def Leer_iris(filename):
    df = pd.read_csv(filename)
    df.replace("Setosa", 1)
    df.replace("Versicolor", 2)
    df.replace("Virginica", 3)
    np_arr = df.to_numpy()
    np_arr = np_arr.T
    temp = [np.ones(np_arr.shape[1]).tolist()]
    for i in range(np_arr.shape[0] - 1):
        temp.append(i.tolist())
    t1, t2, t3 = [], [], []
    for i in np_arr[-1]:
        if i == 1:
            t1.append(1), t2.append(0), t3.append(0)
        elif i == 2:
            t1.append(0), t2.append(1), t3.append(0)
        elif i == 3:
            t1.append(0), t2.append(0), t3.append(1)
    temp.append(t1), temp.append(t2), temp.append(t3)
    data = np.asarray(temp)
    return data

#Funcion para leer Enfermedad_Cardiaca.csv
def Leer_Cardiaca(filename):
    df = pd.read_csv(filename, sep = "\t")
    np_arr = df.to_numpy()
    np_arr = np_arr.T
    temp = [np.ones(np_arr.shape[1]).tolist()]
    for i in range(np_arr[0] - 1):
        temp.append(i.tolist())
    t1, t2 = [], []
    for i in np_arr[-1]:
        if i == 1:
            t1.append(1), t2.append(0)
        else:
            t2.append(1), t1.append(0)
    temp.append(t1), temp.append(t2)
    data = np.asarray(temp)
    return data

#1 - Funcion para leer los datos dado el nombre de archivo (path)
def Leer_Datos(filename, is_iris = True):
    if is_iris:
        data = Leer_iris(filename)
        y_size = 3
    else:
        data = Leer_Cardiaca(filename)
        y_size = 2
    return data, y_size

#2 - Funcion para normalizar los Datos
def Normalizar_Datos(np_arr, ys):
    Media, Desviacion = desviacion_estandar_2(np_arr[1:-ys])
    np_arr[1:-ys] = (np_arr[1:-ys] - Media) / Desviacion
    return np_arr, Media, Desviacion

#3 - Funcion de Accuracy
def Calcular_Accuracy(Y_hat, Y):
    acc = {}
    acc["Setosa"] = np.count_nonzero(Y_hat[0] == 1) / np.count_nonzero(Y[0] == 1)
    acc["Versicolor"] = np.count_nonzero(Y_hat[2] == 1) / np.count_nonzero(Y[1] == 1)
    acc["Virginica"] = np.count_nonzero(Y_hat[1] == 1) / np.count_nonzero(Y[2] == 1)

#4 - Funcion para crear k Folds
def Crear_k_folds(np_arr, k = 3, y_size): # Only works with y = 0 or 1
    unique, counts = [], []
    for i in range(y_size):
        acum = 0
    ##Continuar aqui



    unique, counts = np.unique(np_arr[-1], return_counts = True)
    np_arr = np_arr.T
    dict_unique = {}
    for i in unique:
        dict_unique[i] = []
    for i in np_arr:
        dict_unique[i[-1]].append(i.tolist())
    dict_answer = {}
    for i in range(k):
        dict_answer["k" + str(i)] = []
    for i in range(k - 1):
        for u, c in zip(unique, counts):
            for j in range(int(c / k) * i, int(c / k) * (i + 1)):
                dict_answer["k" + str(i)].append(dict_unique[u][j])
    for u, c in zip(unique, counts):
        for j in range(int(c / k) * (k - 1), c):
            dict_answer["k" + str(k - 1)].append(dict_unique[u][j])
    for i in range(k):
        temp = np.array(dict_answer["k" + str(i)])
        np.random.shuffle(temp)
        dict_answer["k" + str(i)] = temp.T
    return dict_answer

#5 - Funcion Sigmoidal
def Sigmoidal(X, theta):
    return 1/(1 + np.exp(-np.dot(theta.T, X) + eps) + eps)
