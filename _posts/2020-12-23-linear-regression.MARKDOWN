---
layout: post
title: "Machine Learning Algorithms: Linear Regression"
---

Hello! Welcome to this post where we talk about a popular machine learning algorithm called Linear Regression. We'll go through what it is and write it ourselves using python. 

#### What is Linear Regression and how does it work?

The term "regression" in machine learning refers to predicting a certain value. This could be predicting stock prices, house prices and more. 

Linear Regression is a regression technique where the inputs of a dataset are mapped to the outputs of the dataset by applying it to the formula of the **line of best** fit.

![Alt Text](/assets/linear-regression.png)


The general formula for a linear line is **y = mx + c** where x is the input and y is the output. **m** is known as the gradient and **c** is known as the y-intercept. **b** is also used to represent the y-intercept. 

The line of best fit is a line which best expresses a certain dataset. 

The formula to find the gradient (**m**) of the line of best fit is as follows:

![Alt Text](/assets/best-fit-slope.png) 

The formula to find the y-intercept is as follows

![Alt Text](/assets/best-fit-y-intercept.png) 

Where **x** is the list of inputs of the dataset and **y** is the list of outputs

NOTE: Any variable with a line on top of it means you find the mean of that variable

#### Our regression problem

The simple problem we will solve is predicting the price of a house given the number of bedrooms it has. We will just be using a fake dataset. With our dataset, the price would roughly be 100,000 times the number of bedrooms.

#### The Code

```python
class linear_regression:
  def __init__(self):
    self.m = 1
    self.c = 1

  def predict(self, x):
    return self.m * x + self.c

  def mean(self, arr):
    return sum(arr) / len(arr)

  def fit(self, x, y):
    #calculating gradient
    x_mul_y = [_x * _y for _x, _y in zip(x, y)]
    x_squared = [_x ** 2 for _x in x]
    self.m = (((self.mean(x) * self.mean(y)) - self.mean(x_mul_y))) / (self.mean(x) ** 2 - self.mean(x_squared))

    #calculating y-intercept
    self.c = y[0] - self.m * x[0]


#dataset
num_of_bedrooms = [1, 2, 3, 4, 5]
price = [100001, 200001, 310000, 420000, 500000]

l = linear_regression()
l.fit(num_of_bedrooms, price)
print (l.predict(1), l.predict(2), l.predict(10), l.predict(6))

#OUTPUT: 100001.0 202000.7 1017998.3 609999.5
```
