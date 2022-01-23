import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



#lets import the dataset
data_df = pd.read_csv("Auto insurance dataset.csv")


data_df.head()

#lets check the dependies of the variables
plt.scatter(data_df.X, data_df.Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# this is basically following a linear relation. we can fit the dataset by a line which is usually done by Linear regression using Scikit learn . But in this notebook it is done without the sklearn


from random import seed
from random import randrange


#lets make a function to split the data into train and test set
def split(data, test_split_ratio):
    test_df = pd.DataFrame()
    data_size= data.shape[0]
    test_size= test_split_ratio *data_size
    train_df = data.copy()
    while (len(test_df)<(test_size)):
        indexes = randrange(data_size)
        
        test_df = test_df.append(data.iloc[indexes])
        
        train_df = train_df.drop(train_df.index[[indexes]])        
    return train_df, test_df


train_df, test_df = split(data_df, 0.2)
train_df.shape
test_df.shape


# defining the gradient descent function
def gradient_descent(w,b,data, learning_rate):
    dw= 0
    db=0
    for i in range(len(data)):
        x= data.iloc[i].X
        y= data.iloc[i].Y
        dw += -(2/len(data))*x*(y-(w*x+b))
        db += -(2/len(data))*(y-(w*x+b))
    w = w - dw * learning_rate
    b= b - db* learning_rate
    return w,b

#prediction
def predict(w,b,data):
    prediction=[]
    for i in range(len(data)):
        x= data.iloc[i].X
        prediction.append(w*x+b)
    return prediction

#rmse function
def rmse(w, b, data):
    error=0
    for i in range(len(data)):
        x= data.iloc[i].X
        y= data.iloc[i].Y
        error += (y- (w*x +b))**2
    return sqrt(error/float(len(data)))


#for trainset
max_iter =500
lr= 0.00001
w=0
b=0
for i in range(1000):
    w,b=gradient_descent(w,b,train_df, lr)
print("weight:", w)
print("bias: ", b)

rmse(w, b, train_df)

rmse(w, b, test_df)


plt.scatter(train_df.X, train_df.Y, color= 'Black');
plt.plot(train_df.X, predict(w, b, train_df), color = "blue");


# This is the linear regression without using numpy library and sklearn
