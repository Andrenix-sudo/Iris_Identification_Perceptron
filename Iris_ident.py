from matplotlib import pyplot as plt 
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriviative(x):
    return sigmoid(x) * (1 - sigmoid(x))


#slope of cost function is cost * sigmoid derviative
# learning rate  = 0.2
# derivative of the weights  = input values
# derivative of Bias = 1


# updating weights: weight_1 - (learning rate * slope of cost fucntion * input_1)
# bias - (learning rate * slope of the cost function * 1)

# each point is Sepal Lenght (cm), Sepal Weidth (cm) and type 
# type 0 = Iris-setosa & type 1 = Iris-veriscolor
data = [[ 5.1,  3.5,  0 ],
       [ 4.9,  3. ,  0 ],
       [ 4.7,  3.2,  0 ],
       [ 4.6,  3.1,  0 ],
       [ 5. ,  3.6,  0 ],
       [ 5.4,  3.9,  0 ],
       [ 4.6,  3.4,  0 ],
       [ 5. ,  3.4,  0 ],
       [ 4.4,  2.9,  0 ],
       [ 4.9,  3.1,  0 ],
       [ 5.4,  3.7,  0 ],
       [ 4.8,  3.4,  0 ],
       [ 4.8,  3. ,  0 ],
       [ 4.3,  3. ,  0 ],
       [ 5.8,  4. ,  0 ],
       [ 5.7,  4.4,  0 ],
       [ 5.4,  3.9,  0 ],
       [ 5.1,  3.5,  0 ],
       [ 5.7,  3.8,  0 ],
       [ 5.1,  3.8,  0 ],
       [ 7. ,  3.2,  1 ],
       [ 6.4,  3.2,  1 ],
       [ 6.9,  3.1,  1 ],
       [ 5.5,  2.3,  1 ],
       [ 6.5,  2.8,  1 ],
       [ 5.7,  2.8,  1 ],
       [ 6.3,  3.3,  1 ],
       [ 4.9,  2.4,  1 ],
       [ 6.6,  2.9,  1 ],
       [ 5.2,  2.7,  1 ],
       [ 5. ,  2. ,  1 ],
       [ 5.9,  3. ,  1 ],
       [ 6. ,  2.2,  1 ],
       [ 6.1,  2.9,  1 ],
       [ 5.6,  2.9,  1 ],
       [ 6.7,  3.1,  1 ],
       [ 5.6,  3. ,  1 ],
       [ 5.8,  2.7,  1 ],
       [ 6.2,  2.2,  1 ],
       [ 5.6,  2.5,  1 ],
       [ 5.9,  3.2,  1 ],
       [ 6.1,  2.8,  1 ],
       [ 6.3,  2.5,  1 ],
       [ 6.1,  2.8,  1 ],
       [ 6.4,  2.9,  1 ]]

mystery_flower = [6.6, 3]

#plot the scatterpoint data

plt.axis([0, 10, 0, 6])
plt.grid()
for i in range(len(data)):
    point = data[i]
    color = "r"
    if point[2] == 0:
        color = "b"
    plt.scatter(point[0], point[1], c=color)
plt.show()


# Training the data
learning_rate = 0.2
costs = [] # this is the error or the loss from the prediction

# randomly generating weights and bias
weight_1 = np.random.randn()
weight_2 = np.random.randn()
bias = np.random.randn()

for i in range(100000000):
    random_index = np.random.randint(len(data))
    point = data[random_index]
    
    # calculating the weights, inputs and bais for the sigmoid (activiation function)
    z = point[0] * weight_1 + point[1] * weight_2 + bias
    # converts the results of z into a number between 0 and 1
    prediction = sigmoid(z)

    target = point[2] # this is the expected result

    #cost/error calculation. Determins the error as the square of the delta of prediction and target 
    cost = np.square(prediction - target) 

    #derivative of the cost/error calulation. Determins how far off the cost/error calculation is
    dcost_prediction = 2 * (prediction - target)
    # derivative of the sigmoid at z. Determins how far off the sigmoid prediction is
    dprediction_dz = sigmoid_deriviative(z) 

    # derivative of weights is the input value because they are constant
    # derivative of bias is 1
    dz_dweight_1 = point[0]
    dz_dweight_2 = point[1]
    dz_dbais = 1

    #slope of the cost/error calculation - Not sure why this is needed
    dcost_dz = dcost_prediction * dprediction_dz

    # slope of the weights & bias - not sure why this is needed
    dcost_dw1 = dcost_dz * dz_dweight_1
    dcost_dw2 = dcost_dz * dz_dweight_2
    dcost_bias = dcost_dz * dz_dbais

    # getting new weights by multiplying the learning rate by the slop of the weights/bias

    weight_1 = learning_rate * dcost_dw1
    weight_2 = learning_rate * dcost_dw2
    bias = learning_rate * dcost_bias


# for use after training on the data
test_data = [[ 5.4,  3.4,  0. ],
           [ 4.6,  3.6,  0. ],
           [ 5.1,  3.3,  0. ],
           [ 4.8,  3.4,  0. ],
           [ 5. ,  3. ,  0. ],
           [ 5. ,  3.4,  0. ],
           [ 5.2,  3.5,  0. ],
           [ 5.2,  3.4,  0. ],
           [ 4.7,  3.2,  0. ],
           [ 4.8,  3.1,  0. ],
           [ 5.4,  3.4,  0. ],
           [ 5.2,  4.1,  0. ],
           [ 5.5,  4.2,  0. ],
           [ 4.9,  3.1,  0. ],
           [ 5. ,  3.2,  0. ],
           [ 5.5,  3.5,  0. ],
           [ 4.9,  3.1,  0. ],
           [ 4.4,  3. ,  0. ],
           [ 5.1,  3.4,  0. ],
           [ 5. ,  3.5,  0. ],
           [ 4.5,  2.3,  0. ],
           [ 4.4,  3.2,  0. ],
           [ 5. ,  3.5,  0. ],
           [ 5.1,  3.8,  0. ],
           [ 4.8,  3. ,  0. ],
           [ 5.1,  3.8,  0. ],
           [ 4.6,  3.2,  0. ],
           [ 5.3,  3.7,  0. ],
           [ 5. ,  3.3,  0. ],
           [ 6.8,  2.8,  1. ],
           [ 6.7,  3. ,  1. ],
           [ 6. ,  2.9,  1. ],
           [ 5.7,  2.6,  1. ],
           [ 5.5,  2.4,  1. ],
           [ 5.5,  2.4,  1. ],
           [ 5.8,  2.7,  1. ],
           [ 6. ,  2.7,  1. ],
           [ 5.4,  3. ,  1. ],
           [ 6. ,  3.4,  1. ],
           [ 6.7,  3.1,  1. ],
           [ 6.3,  2.3,  1. ],
           [ 5.6,  3. ,  1. ],
           [ 5.5,  2.5,  1. ],
           [ 5.5,  2.6,  1. ],
           [ 6.1,  3. ,  1. ],
           [ 5.8,  2.6,  1. ],
           [ 5. ,  2.3,  1. ],
           [ 5.6,  2.7,  1. ],
           [ 5.7,  3. ,  1. ],
           [ 5.7,  2.9,  1. ],
           [ 6.2,  2.9,  1. ],
           [ 5.1,  2.5,  1. ],
           [ 5.7,  2.8,  1. ]]

for i in range(len(test_data)):
    point = test_data[i]
    print(point)

    z = point[0] * weight_1 + point[1] * weight_2 * bias
    prediction = sigmoid(z)

    print("prediction : {}".format(prediction))

# function for predicting the flower

def guess_flower(sepal_length, sepal_width):
    z = sepal_length * weight_1 + sepal_width * weight_1 * bias
    prediction = sigmoid(z)
    if prediction < 0.5:
        print("Iris-setosa")
    else:
        print("Iris-setosa")


guess_flower(4.8, 3.0)

guess_flower(6.7, 3.0)


guess_flower(5.1, 3.7)