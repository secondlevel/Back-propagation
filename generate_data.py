import numpy as np
import matplotlib.pyplot as plt

def generator_linear(n=100):

    pts = np.random.uniform(0, 1 , (n, 2))
    inputs, labels = [],[] 
    for pt in pts:
        distance = (pt[0]-pt[1])/1.414
        inputs.append([pt[0],pt[1]])
        if pt[0]>pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    inputs = np.array(inputs)
    labels = np.array(labels).reshape(n , 1)

    return inputs, labels

def generator_XOR_easy(n = 11):
    inputs,labels = [],[] 
    for i in range(n):

        prod_number = 11/n
        inputs.append([0.1*prod_number*i,0.1*prod_number*i])
        labels.append(0)

        if 0.1*prod_number*i==0.5:
            continue

        inputs.append([0.1*prod_number*i,1-0.1*prod_number*i])
        labels.append(1)
    
    inputs = np.array(inputs)
    labels = np.array(labels).reshape(n*2-1 , 1)

    return inputs, labels

def plot_data(inputs,labels):
    for i in range(inputs.shape[0]):
        if labels[i][0]==0:
            plt.plot(inputs[i][0],inputs[i][1],'o',color="blue")
        else:
            plt.plot(inputs[i][0],inputs[i][1],'o',color="red")
        plt.show()

# if __name__ =="__main__":

#     a = generator_linear(1000)
#     b = generator_XOR_easy(11)

#     print("linear classifier input coordinate:", a[0].shape)
#     print("linear classifier input label:", a[1].shape)
#     print("XOR classifier input coordinate:", b[0].shape)
#     print("XOR classifier input label:", b[1].shape)
