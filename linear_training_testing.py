import numpy as np
from generate_data import generator_linear,generator_XOR_easy
import matplotlib.pyplot as plt
import collections
import pandas as pd

def plot_training_result(X,Y,pred_label,epochs):

    plt.ion()
    plt.cla()
    plt.suptitle("Training Result(Epochs = "+str(epochs)+")",fontsize=15)

    plt.subplot(1,2,1)
    plt.title("Ground Truth",fontsize=14)
    plt.scatter(np.transpose(X)[:, 0], np.transpose(X)[:, 1], c=np.transpose(Y).reshape(np.transpose(Y).shape[0],), s=100, lw=0, cmap='coolwarm')

    plt.subplot(1,2,2)
    plt.title("Predict Result",fontsize=14)
    plt.scatter(np.transpose(X)[:, 0], np.transpose(X)[:, 1], c=np.transpose(pred_label).reshape(np.transpose(pred_label).shape[0],), s=100, lw=0, cmap='coolwarm')
    plt.show()
    plt.pause(1e-4)
    plt.ioff()

def plot_testing_result(X,Y,pred_label):

    #用了之後，順序性會消失
    plt.figure()

    plt.suptitle("Testing Result",fontsize=15)

    plt.subplot(1,2,1)
    plt.title("Ground Truth",fontsize=14)
    plt.scatter(X[:, 0], X[:, 1], c=Y.reshape(Y.shape[0],), s=100, lw=0, cmap='coolwarm')

    plt.subplot(1,2,2)
    plt.title("Predict Result",fontsize=14)
    plt.scatter(X[:, 0], X[:, 1], c=np.transpose(pred_label).reshape(np.transpose(pred_label).shape[0],), s=100, lw=0, cmap='coolwarm')
    plt.show()

def plot_training_curve(accuracy_history,loss_history):
    
    #用了之後，順序性會消失
    plt.close()
    plt.figure()

    plt.suptitle("Training Curve",fontsize=15)

    plt.subplot(1,2,1)
    plt.title("Accuracy Curve",fontsize=14)
    plt.plot(accuracy_history,c='r')
    
    plt.xlabel("Epochs")

    plt.subplot(1,2,2)
    plt.title("Loss Curve",fontsize=14)
    plt.plot(loss_history,c='b')
    plt.xlabel("Epochs")

    # plt.show()

def get_loss(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return Y_hat_,(Y_hat_ == Y).all(axis=0).mean()

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def normal(Z):
    return Z

def normal_backward(dA, Z):
    return dA

def init_layers(nn_architecture, seed = 99):
    # 設定隨機種子
    np.random.seed(seed)
    # neural network的數量
    number_of_layers = len(nn_architecture)
    # 當前權重W0 W1 及誤差b1 b2的儲存地方
    params_values = {}
    
    # 開始隨機設置每一層的權重跟誤差，初始設定的值需要小一點會比較好
    for idx, layer in enumerate(nn_architecture):
        # W1 W2 b1 b2的數字，從1開始
        layer_idx = idx + 1
        
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values

def single_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        activation_func = normal
        
    return activation_func(Z_curr), Z_curr

def forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory

def single_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        backward_activation_func = normal_backward
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values

def update_model_weight_bias(params_values, grads_values, nn_architecture, learning_rate):

    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values

def train(X, Y, nn_architecture, epochs, learning_rate, verbose=True, callback=None):
    params_values = init_layers(nn_architecture, 2)
    loss_history = []
    accuracy_history = []

    for i in range(epochs):    

        Y_hat, cashe = forward_propagation(X, params_values, nn_architecture)
        
        loss = get_loss(Y_hat, Y)
        loss_history.append(loss)
        pred_label,accuracy = get_accuracy(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        grads_values = backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update_model_weight_bias(params_values, grads_values, nn_architecture, learning_rate)
        
        if(i % 10 == 0):
            if(verbose):

                print("Epoch: {:06}  Loss: {:.6f}  Accuracy: {:.6f}".format(i, loss, accuracy))
                # plot_training_result(X,Y,pred_label,i)
            
            if(callback is not None):
                callback(i, params_values)

    plot_training_curve(accuracy_history,loss_history)        
    return params_values

if __name__ == "__main__":

    data_number = 1000

    x, y = generator_linear(n = data_number)
    x = x.astype('float32')
    y = y.astype('float32').reshape(data_number,)

    TwoLayerNet = [
    {"input_dim": 2, "output_dim": 20, "activation": "sigmoid"},
    {"input_dim": 20, "output_dim": 10, "activation": "sigmoid"},
    {"input_dim": 10, "output_dim": 1, "activation": "sigmoid"},
    ]

    # plt.scatter(x[:,0],x[:,1],c = y,s=100,lw=0,cmap='coolwarm')
    # plt.show()

    # print(x.shape)
    # print(y.shape)

    #training 
    params_values = train(np.transpose(x), np.transpose(y.reshape((y.shape[0], 1))), TwoLayerNet, 6000, 5e-1)

    #testing
    data_number = 1000

    x_test, y_test = generator_linear(n = data_number)
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32').reshape(data_number,)
    
    y_pred, _ = forward_propagation(np.transpose(x_test), params_values, TwoLayerNet)
    pred_label_test,acc_test = get_accuracy(y_pred, np.transpose(y_test.reshape((y_test.shape[0], 1))))

    print("----------------------------------------------------------------------------------")
    data_point_truth=collections.Counter(y_test.reshape(y_test.shape[0],))
    data_point_prediction=collections.Counter(np.transpose(pred_label_test).reshape(np.transpose(pred_label_test).shape[0],))
    data_point_truth = dict(data_point_truth)
    data_point_prediction = dict(data_point_prediction)
    
    if len(data_point_prediction)==2:
        print("Ground Truth 的紅點數量有:",data_point_truth[0.0]," ,藍點數量有",data_point_truth[1.0])
        print("prediction 的紅點數量有:",data_point_prediction[0.0]," ,藍點數量有",data_point_prediction[1.0],"\n")
        testing_confusion_matrix = pd.crosstab(y_test, np.transpose(pred_label_test).reshape(np.transpose(pred_label_test).shape[0],),rownames=['Ground Truth'], colnames=['Prediction'])
        print("\nTesting accuracy: {:.6f}\n".format(acc_test))
        print("Testing Confusion Matrix:\n\n",testing_confusion_matrix)
        print("Ground Truth:",y_test)
        print("Predict Result:",np.transpose(pred_label_test).reshape(np.transpose(pred_label_test).shape[0],))
        plot_testing_result(x_test,y_test,pred_label_test)