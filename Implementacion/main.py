from funciones import *

def main():
    kernels_SVM = ["linear", "poly", "rbf", "sigmoid"]
    C_value = [1, 100, 500, 1000]
    iterations_values = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    learning_rates = [0.7, 0.4, 0.1, 0.07,0.05, 0.01]
    datasets = ["../Datos/Enfermedad_Cardiaca.csv", "../Datos/iris.csv"]
    #extraccion de datos
    EC_dataset, y_ec = Leer_Datos(datasets[0], False)
    iris_dataset, y_ir = Leer_Datos(datasets[1], True)
    #Normalizacion
    EC_dataset, __,__ = Normalizar_Datos(EC_dataset, y_ec)
    iris_dataset, __,__ = Normalizar_Datos(iris_dataset, y_ir)
    #aplicando kfolds
    kfoldEC,__ = Crear_k_folds(EC_dataset, y_ec)
    kfoldir,__ = Crear_k_folds(iris_dataset, y_ir)
    #Arquitectura de las redes en en EC
    arq_MLP_EC = [14, 15, 7, 2]
    arq_MLP_ir = [5, 10, 5, 3]

    
    #Experimento 2
    EC_dataset, y_ec = Leer_SVM_ec(datasets[0]), 2
    iris_dataset, y_ir = Leer_SVM_ir(datasets[1]), 3
    #Normalizacion
    EC_dataset, __,__ = Normalizar_Datos(EC_dataset, y_ec)
    iris_dataset, __,__ = Normalizar_Datos(iris_dataset, y_ir)
    #aplicando kfolds
    kfoldEC,__ = Crear_k_folds_SVM(EC_dataset, y_ec)
    kfoldir,__ = Crear_k_folds_SVM(iris_dataset, y_ir)

    print("Experimento 2")
    #EC
    print("Resultados Enfermedad Cardiaca")
    for i in kernels_SVM:
        it_list = []
        for j in C_value:
            acc = 0
            for f in range(y_ec ** 3):
                train, test = kfoldEC["f" + str(f) + "-train"], kfoldEC["f" + str(f) + "-test"]
                X_train, Y_train = train[:-1].T, train[-1:]
                Y_train = np.reshape(Y_train, Y_train.shape[1])
                X_test, Y_test = test[:-1].T, test[-1:]
                Y_test = np.reshape(Y_test, Y_test.shape[1])
                acc += SVM_tester(i, j, X_train, Y_train, X_test, Y_test)
            acc /= y_ec ** 3
            it_list.append(acc)
        answer_line = "\\textbf{" + i + "} & "
        for i in it_list:
            answer_line += str(round(i, 4)) + " & "
        answer_line += "\\\\"
        print(answer_line)

    #iris
    print("Resultados iris")
    for i in kernels_SVM:
        it_list = []
        for j in C_value:
            acc = 0
            for f in range(y_ir ** 3):
                train, test = kfoldir["f" + str(f) + "-train"], kfoldir["f" + str(f) + "-test"]
                X_train, Y_train = train[:-1].T, train[-1:]
                Y_train = np.reshape(Y_train, Y_train.shape[1])
                X_test, Y_test = test[:-1].T, test[-1:]
                Y_test = np.reshape(Y_test, Y_test.shape[1])
                acc += SVM_tester(i, j, X_train, Y_train, X_test, Y_test)
            acc /= y_ir ** 3
            it_list.append(acc)
        answer_line = "\\textbf{" + i + "} & "
        for i in it_list:
            answer_line += str(round(i, 4)) + " & "
        answer_line += "\\\\"
        print(answer_line)
    return True
    print("Resultados Experimento 1")
    #EC
    print("Resultados Enfermedad Cardiaca")
    num_capas = len(arq_MLP_EC)
    for i in iterations_values:
        it_list = []
        for j in learning_rates:
            acc = 0
            for f in range(y_ec ** 3):
                train, test = kfoldEC["f" + str(f) + "-train"], kfoldEC["f" + str(f) + "-test"]
                #print(train.shape)
                X_train, Y_train = train[:-y_ec], train[-y_ec:]
                X_test, Y_test = test[:-y_ec], test[-y_ec:]
                #temp_params = gen_params(arq_MLP_EC)
                temp_params = {}
                W_dic = {}
                temp_params["a0"] = X_train
                W_dic = gen_weight_dic(arq_MLP_EC)
                temp_params, W_dic = Gradiente_Descendiente(temp_params, W_dic, Y_train, len(arq_MLP_EC), j , i)
                temp_params["a0"] = X_test
                temp_params = Forward(temp_params, W_dic, num_capas)
                Y_hat = temp_params["a3"]
                acc += Calcular_Accuracy(Y_hat, Y_test)
            acc /= y_ec ** 3
            it_list.append(acc)
        answer_line = "\\textbf{" + str(i) + "} & "
        for i in it_list:
            answer_line += str(round(i, 4)) + " & "
        answer_line += "\\\\"
        print(answer_line)
    #return True
    
    #Experimento 1
    print("Resultados Experimento 1")
    #iris
    print("Resultados iris")
    num_capas = len(arq_MLP_ir)
    for i in iterations_values:
        it_list = []
        for j in learning_rates:
            acc = 0
            for f in range(y_ir ** 3):
                train, test = kfoldir["f" + str(f) + "-train"], kfoldir["f" + str(f) + "-test"]
                #print(train.shape)
                X_train, Y_train = train[:-y_ir], train[-y_ir:]
                X_test, Y_test = test[:-y_ir], test[-y_ir:]
                #temp_params = gen_params(arq_MLP_ir)
                temp_params = {}
                W_dic = {}
                temp_params["a0"] = X_train
                #print(X_train.shape)
                W_dic = gen_weight_dic(arq_MLP_ir)
                temp_params, W_dic = Gradiente_Descendiente(temp_params, W_dic, Y_train, len(arq_MLP_ir), j , i)
                temp_params["a0"] = X_test
                temp_params = Forward(temp_params, W_dic, num_capas)
                Y_hat = temp_params["a3"]
                acc += Calcular_Accuracy(Y_hat, Y_test)
            acc /= y_ir ** 3
            it_list.append(acc)
        answer_line = "\\textbf{" + str(i) + "} & "
        for i in it_list:
            answer_line += str(round(i, 4)) + " & "
        answer_line += "\\\\"
        print(answer_line)
    




if __name__ == "__main__":
    main()
