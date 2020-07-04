#Hecho por Ricardo Manuel Lazo Vásquez - Universidad Catolica San Pablo - TheReverseWasp (GitHub)
#Esta implementacion busca implementar un Perceptron Multicapa ademas de una Support Vector Machine

import numpy as np
import pandas as pd
import random

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from complementos import *

#constante para compilacion de logaritmos
eps = 1e-10

#Funcion para leer iris dataset
#Solo funciona con iris.csv asi que si se quiere atacar otro dataset modificar esta o crear otra función
def Leer_iris(filename):
    #print("entra")
    df = pd.read_csv(filename)
    df.replace("Setosa", 1)
    df.replace("Versicolor", 2)
    df.replace("Virginica", 3)
    np_arr = df.to_numpy()
    np_arr = np_arr.T
    temp = [np.ones(np_arr.shape[1]).tolist()]
    for i in np_arr[:-1]:
        temp.append(i.tolist())
    t1, t2, t3 = [], [], []
    for i in np_arr[-1]:
        if i == 0:
            t1.append(1), t2.append(0), t3.append(0)
        elif i == 1:
            t1.append(0), t2.append(1), t3.append(0)
        elif i == 2:
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
    for i in np_arr[:-1]:
        temp.append(i.tolist())
    t1, t2 = [], []
    for i in np_arr[-1]:
        if i == 0:
            t1.append(1), t2.append(0)
        else:
            t2.append(1), t1.append(0)
    temp.append(t1), temp.append(t2)
    data = np.asarray(temp)
    return data


#Opcional Leer SVM
def Leer_SVM_ec(filename):
    df = pd.read_csv(filename, sep = "\t")
    np_arr = df.to_numpy()
    np_arr = np_arr.T
    return np_arr


def Leer_SVM_ir(filename):
    df = pd.read_csv(filename)
    np_arr = df.to_numpy()
    np_arr = np_arr.T
    return np_arr

#1 - Funcion para leer los datos dado el nombre de archivo (path)
def Leer_Datos(filename, is_iris):
    if is_iris:
        data = Leer_iris(filename)
        y_size = 3
    else:
        data = Leer_Cardiaca(filename)
        y_size = 2
    data = data.astype("float128")
    return data, y_size

#2 - Funcion para normalizar los Datos
def Normalizar_Datos(np_arr, ys):
    Media, Desviacion = desviacion_estandar_2(np_arr[1:-ys])
    np_arr[1:-ys] = (np_arr[1:-ys] - Media) / Desviacion
    return np_arr, Media, Desviacion

#3 - Funcion de Accuracy
def Calcular_Accuracy(Y_hat, Y):
    acc = 0
    dims, test_cases = Y.shape
    for i in range(dims):
        for j in range(test_cases):
            if Y[i][j] == Y_hat[i][j]:
                acc +=1
    acc /= dims * test_cases
    return acc

#4 - Funcion para crear k Folds
def Crear_k_folds(np_arr, y_size, k = 3): # Only works with y = 0 or 1
    #print(np_arr.shape)
    groups = {}
    for i in range(y_size):
        groups[i] = []
    for i in np_arr.T:
        for j in range(-y_size, -1):
            if i[j] == 1:
                groups[y_size + j].append(i)
    for i in range(y_size):
        groups[i] = np.asarray(groups[i])
    #print(str(groups))
   # print("groups 0", groups[0].shape)
    for i in range(y_size):
        random.shuffle(groups[i])
    #print(groups[0].shape)
    #print("Lol")
    dict_answer = {}
    fold_max_size = y_size ** k
    for i in range(fold_max_size):
        selection_matrix = k_folds_helper(k, y_size, i)
        dict_answer["f" + str(i)+"-train"] = []
        dict_answer["f" + str(i)+"-test"] = []
        for j in range(y_size):
            group_size = len(groups[j])
            fold_size = group_size / k
            for k in range(group_size):
                if k > selection_matrix[j] * fold_size and k < selection_matrix[j] * fold_size + fold_size:
                    #test
                    dict_answer["f" + str(i) + "-test"].append(groups[j][k])
                else:
                    #train
                    dict_answer["f" + str(i) + "-train"].append(groups[j][k])
        dict_answer["f" + str(i) + "-train"] = np.asarray(dict_answer["f" + str(i) + "-train"]).T
        dict_answer["f" + str(i) + "-test"] = np.asarray(dict_answer["f" + str(i) + "-test"]).T
    return dict_answer, fold_max_size

def Crear_k_folds_SVM(np_arr, y_size, k = 3): # Only works with y = 0 or 1
    #print(np_arr.shape)
    groups = {}
    for i in range(y_size):
        groups[i] = []
    print("size", np_arr.T.shape)
    for i in np_arr.T:
        groups[i[-1]].append(i)
    for i in range(y_size):
        groups[i] = np.asarray(groups[i])
    #print(str(groups))
   # print("groups 0", groups[0].shape)
    for i in range(y_size):
        random.shuffle(groups[i])
    #print(groups[0].shape)
    #print("Lol")
    dict_answer = {}
    fold_max_size = y_size ** k
    for i in range(fold_max_size):
        selection_matrix = k_folds_helper(k, y_size, i)
        dict_answer["f" + str(i)+"-train"] = []
        dict_answer["f" + str(i)+"-test"] = []
        for j in range(y_size):
            group_size = len(groups[j])
            fold_size = group_size / k
            for k in range(group_size):
                if k > selection_matrix[j] * fold_size and k < selection_matrix[j] * fold_size + fold_size:
                    #test
                    dict_answer["f" + str(i) + "-test"].append(groups[j][k])
                else:
                    #train
                    dict_answer["f" + str(i) + "-train"].append(groups[j][k])
        dict_answer["f" + str(i) + "-train"] = np.asarray(dict_answer["f" + str(i) + "-train"]).T
        dict_answer["f" + str(i) + "-test"] = np.asarray(dict_answer["f" + str(i) + "-test"]).T
    return dict_answer, fold_max_size


#5 - Funcion Sigmoidal
def Sigmoidal(Z):
    answer = np.round(1/(1 + np.exp(-Z)), 25)
    #print (answer)
    return answer

#6 Funcion Costo General
def Calcular_Funcion_Costo(Y_hat, Y):
    return np.sum((Y_hat - Y) ** 2) / (2 * Y.shape[1])

#7 Derivada de Sigmoide
def dS(Y_hat):
    return Y_hat * (1 - Y_hat)

#8 Forward Propagation
def Forward(temp_params, W_dic, num_capas):
    for i in range(1, num_capas):
        temp_params["z" + str(i)] = np.dot(np.concatenate((W_dic["w" + str(i)].T, W_dic["b" + str(i)]), axis = 1), temp_params["a" + str(i - 1)]) 
        rows, cols = W_dic["w" + str(i)].shape[1], temp_params["a" + str(i - 1)].shape[1]
        if i != num_capas:
            temp_params["a" + str(i)] = np.ones((rows + 1, cols))
            temp_params["a" + str(i)][1:] = Sigmoidal(temp_params["z" + str(i)])
        else:
            temp_params["a" + str(i)] = np.ones((rows, cols))
            temp_params["a" + str(i)] = Sigmoidal(temp_params["z" + str(i)])
    return temp_params

#9 Backward Propagation
def Backward(temp_params, W_dic, Y, num_capas, learning_rate):
    dLaf = np.sum(-(Y - temp_params["a" + str(num_capas - 1)]), axis = 1, keepdims = 1)
    T = dLaf
    for i in range(num_capas - 1, 0, -1):
        dadz = dS(temp_params["a" + str(i)])
        #print(T.shape, dadz.shape)
        R = np.dot(T.T, dadz)
        T = np.dot(temp_params["a" + str(i - 1)], R.T)
        #print(T.shape)
        dW = T[1:]
        db = T[0]
        #db = np.reshape(db, (db.shape[0], 1))
        #print (dW.shape, W_dic["w" + str(i)].shape)
        #print ("db", db.shape, W_dic["b" + str(i)].shape)
        W_dic["w" + str(i)] -= learning_rate * dW
        W_dic["b" + str(i)] -= learning_rate * db
    return W_dic

#10 Gradiente descendiente
def Gradiente_Descendiente(temp_params, W_dic, Y, num_capas, learning_rate = 0.0001, num_iteraciones = 500):
    #print("lr: ", learning_rate)
    for i in range(num_iteraciones):
        temp_params = Forward(temp_params, W_dic, num_capas)
        W_dic = Backward(temp_params, W_dic, Y, num_capas, learning_rate)
    return temp_params, W_dic

#Experimento 2 SVM
def SVM_tester(kernel_SVM, C_value, X_train, Y_train, X_test, Y_test):
    my_svm = SVC(kernel = kernel_SVM, C = C_value, random_state = 0)
    my_svm.fit(X_train, Y_train)
    Y_hat = my_svm.predict(X_test)
    accuracy_result = accuracy_score(Y_test, Y_hat)
    return accuracy_result
