import time

import matplotlib.pyplot as plt
import numpy as np
# import h5py
import scipy
from scipy import ndimage

from lr_utils import load_dataset

# 载入数据，HDF5文件类型，其中有两个。一个是训练集，一个是测试集 其中load_dataset()函数是从lr_utils.py中导入的
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#得到训练集和测试集的数量，以及num_px是图片的长和宽度，单位是像素点
# 其中train_set_x_orig.shape  是 209 64 64 3

m_train = train_set_x_orig.shape[0]  # 这个是训练样本的数量  此程序是209张
m_test = test_set_x_orig.shape[0]  # 这个是测试样本的数量  此程序是50张
num_px = train_set_x_orig.shape[1]  # 这个是每个图片的像素点

#将图像原本是[64,64,3]的列表变成[64*64*3，1]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],
                                               -1).T  #这里用了reshape()函数-1就是我们不知道train_set_x_orig的列数
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#将数据进行标准化，也就是说，平均一下。
train_set_x = train_set_x_flatten / 255.  #将数据集中的每一行除以255（像素通道的最大值）
test_set_x = test_set_x_flatten/255.

#定义激活函数
def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s

#将w和b初始化为0的函数
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))  #这里要注意w一定要是一个向量才行，不能直接等于0，那样的话就是一个浮点数
    b = 0

    #确保w是一个列向量，并且维度是确定，确保b是float或者int
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b



#正向传播和反向传播算法，计算sigmodi值，计算损失函数的数值
def propagate(w, b, X, Y):
    m = X.shape[1]
    # 正向传播计算cost 的数值
    A = sigmoid(np.dot(w.T, X) + b)  # 计算激活函数的值
    cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / (-m)  # 计算损失函数的值损失函数是-(ylog(A)+(1-y)log(1-A))
    # 反向传播求导数
    dw = np.dot(X, (A - Y).T) / m  #这一点很重要，用向量化的方式很简单的计算了成本函数对w的偏导
    db = np.sum(A - Y) / m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost


w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)

#优化算法
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = np.dot(w.T, X)
    for i in range(A.shape[1]):

        # Convert probabilities a[0,i] to actual predictions p[0,i]
        if (A[0, i] > 0.5):
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


tic = time.process_time()
num_iterations = 5000
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = num_iterations, learning_rate = 0.005, print_cost = True)
toc = time.process_time()
print('Use num_iterations of %i, run %f sec ' %(num_iterations, toc - tic))


costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


my_image = "./w2.jpg"

# 预处理自己的图片适应上面的逻辑回归算法
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", 逻辑回归预测的该结果是 \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")