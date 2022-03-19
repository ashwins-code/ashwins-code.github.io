---
layout: post
title: "Linear Algebra Basics"
---


Hello and welcome to this post about the basics of Linear Algebra. I have just begun to read a book called "Mathematics for Machine Learning" by Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong. Linear Algebra is the first bit of mathematics the book talks about. Linear Algebra is the core of many machine learning models. In this post, we will cover what Linear Algebra is, what Vectors and Matrices are and some operations using matrices.


**What is Linear Algebra**

Algebra is a set of rules that is used to manipulate certains objects/symbols. Linear Algebra is similar, in that it is the set of rules to manipulate "vectors".

**What are Vectors**

In general, vectors are objects that can be added together and be multiplied by a scalar to produce an object of the same type.

Vectors are commonly known as an object that specifies a direction along with a magnitude. This is known as a geometric vector. Two of these vectors can be added to give another vector. These vectors can also be multipled by a scalar to give another vector. This fits with our general definition, hence why it is considered a vector.

![Alt Text](/assets/vector.png) 


Other than geometric vectors, polynomials can be considered unusual vectors. Like the geometric vectors, they can be added together (resulting in another polynomial) and can be multiplied by a scalar to produce another polynomial. This is why polynomials can be an instance of a vector, with our general definition in mind.

Most obvious to our general definition, any real number is also a vector.

**What are Matrices**

Matrices are central in Linear Algebra.

A Matrix is a set of numbers arranged in a rectangular shape. A matrix's shape is number of rows x number of columns. A maxtrix which is has just one row or one column is known as a row/column vector. 

In terms of a geometric plane, a matrix is used to map points on the plane to other point on the plane (know as transformations). However, matrices have many more uses such as the calculations used in neural networks.

![Alt Text](/assets/matrix.png) 
<sup>A 2x2 matrix</sup>

In this post, we will represent matrices as such

```
[ 1, 2
  3, 4 ]
```

**Matrix addition**

Matrix addition is very simple. The sum of two matrcies is an element wise sum. 

```
A = [ 1, 2 
      3, 4 ]
B = [ 2, 3
      4, 5 ]

A + B = [ 3, 5 
          7, 9 ]  
```
**Multiplying a matrix by a scalar**

Multiplying a matrix by a scalar is also straightforward. You multiply each element of the matrix by the scalar to produce the new matrix.

```
2 * [ 1, 2
      3, 4 ] 

= [ 2, 4
    6, 8 ]
```

**Matrix Multiplication**

Unlike matrix addition, matrix multiplication is not element wise (element wise multiplication is called the Hadamard product). Instead, it is done like below

![Alt Text](/assets/matmul.png) 

In matrix multiplication, for every row of the left matrix (call this the Ath row), you go through each column of the right matrix (call this the Bth column) and perform a "dot product".

The "dot product" is when you multiply matching members and sum them up. 

```
Dot product of [1, 2, 3] and [2, 3, 4]

(1*2)+(2*3)+(3*4) = 2 + 6 + 12 = 20
```

The result of the dot product is then placed in row A and column B of the resulting matrix.

The resulting matrix should have the same number of rows as the left matrix, and the same number of columns as the right matrix.


```
A = [ 1, 2
      3, 4
      5, 6 ] 

B = [ 1, 2
      3, 4 ]

A * B = [ (1*1)+(2*3), (1*2)+(2*4) 
          (3*1)+(4*3), (3*2)+(4*4)
          (5*1)+(6*3), (5*2)+(6*4) ]

      = [ 7, 10 
         15, 22
         23, 34 ]
```



**Transposing a matrix**

Transposing a matrix is when you take a matrix and make its rows columns and its columns rows. 

```
[ 1 2 
  3 4
  5 6 ]
```

Tranposing this gives

```
[ 1 3 5
  2 4 6 ]
```

Notation: A<sup>T</sup> means the transposed form of A

**Inverting a matrix**

The inverse of a matrix does not always exist. An inverse of a matrix is one that, when it is multiplied by its uninverted form, it produces an "identity matrix"

An identity matrix is a matrix where the diagonal from top left to bottom right is full of 1s. The rest are filled with 0s.

For a 2x2 matrix, the inverted form is as follows

```
A = [ a, b
      c, d ]

Inverted A = (1/ad-bc) * [ d, -b
                          -c, a ]
```

Taking the inverse of a matrix is like taking the reciprocal of a number (reciprocal of x is 1/x)


Notation: A<sup>-1</sup> means the inverse of A.

We use the inverse to be able to divide matrices. Since there is no concept of dividing by a matrix, multiplying by the inverse of a matrix achieves the same result.

Therefore, to get A/B (where A and B are matrices), you do AB<sup>-1</sup>

**Where is all of this used in Machine Learning**

Artificial Neural Networks rely heavily on Calculus and matrices. Matrix multiplication is used for the network to make predictions and Calculus is used to make the network learn.

![Alt Text](/assets/neuralnet.jpeg) 

Another application is in images. An image can be represented as a matrix, since it is just a rectangular structure of pixels. Once it is represented as matrix, it is nothing but an object which we can perform mathematical calculations on. These calculations are what leads to common image operations we see today such as cropping, scaling and adding effects to the image.

Finally, linear algebra is used in recommender systems. At the most basic level, recommender systems calculate how similar something is to your interests are and will recommend them to you if they are similar enough to your interests. The way similaraity is calculated is by using distance measures such as Euclidean distance or dot products, which return the distance between matrices as if they were plotted on a graph.