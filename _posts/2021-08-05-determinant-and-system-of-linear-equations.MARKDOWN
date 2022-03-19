---
layout: post
title: "Linear Algebra: Determinants, matrix invertability and systems of linear equations"
---

Hi! Welcome to this post, where we will cover determinants of a matrix, inverse matrices, how to solve a system of linear equations, using matrices. 

**What is a determinant and what is it for?**

The determinant is a number than can be calculated from a SQUARE matrix. A square matrix means the number of columns and the number of rows are the same.

The determinant is used to calculate the inverse of a matrix, which has uses in solving systems of linear equations and in vector calculus.

**Calculating the determinant**

For a 2x2 matrix, the determinant is calculated as follows: ad - bc

Where the matrix is...

```
[ a b 
  c d ]
```

So the determinant of this matrix...

```
[ 1 2
  3 4 ]
```

would be 1 * 4 - 2 * 3 = -2

To calculate the determinant of a matrix that is larger than 2x2, you do the following

Go through each number in the top row (call this number x)
Cover all the numbers in x's row and column. The remaining, uncovered numbers form a matrix (call this matrix y)
Take the determinant of matrix y and multiply it with x (call this result z).
If x is in an even position in the top row, multiply z by -1, otherwis leave z as it is.
The determinant of the matrix is the sum of all the "z" values 

Let's use an example

```
[ 1 2 3 
  4 5 6
  7 8 9 ]
```

We go through each number of the first row

At 1...

Cover all the the numbers in 1's row and column and find the determinant of the remaining numbers

```
Remaining numbers
[ 5 6
  8 9 ]
```

Determinant = 5 * 9 - 6 * 8 = -3


Then multiply the determinant by the current top row number 

-3 * 1 = -3 (call this z1)


At 2...

The process repeats

```
Remaning numbers
[ 4 6 
  7 9 ]
```

Determinant = 4 * 9 - 7 * 6 = -6

2 * -6 = -12

However, since 2 is at an even position in the top row, we multiply this by -1.

-20 * -1 = 12 (call this z2)

At 3...

```
Remaining numbers
[ 4 5
  7 8 ]
```

Determinant = 4 * 8 - 5 * 7 = -3

3 * -3 = -9 (call this z3)


The determinant of the whole matrix would be z1 + z2 + z3, which in this case, is -3 + 12 + -9 = 0

**Inverse of Matrices**

The inverse of a matrix is like taking the reciprocal of a number (1/n).

When a matrix is multipled by its inverse, the result is an identity matrix (where the left to right diagnol is full of ones, and the rest of the matrix is a 0)

**Calculating the inverse**

For a 2x2 a matrix,

```
[ a b 
  c d ]
```

The inverse is (1/determinant) * [ d -b 
                                  -c  d ]

For a matrix larger than a 2x2...

You first form a matrix of minors from a matrix
-A matrix of minors is formed by 
--initially having an empty matrix the same shape as the matrix (call this matrix x)
--going through each number in the matrix 
--covering up all the numbers in current number's row and column
--calculating the determinant of the uncovered numbers
--placing the result in the same position as the current number in matrix x

Let's call the matrix of minors m

After that, you apply the cofactors, which is simply done by multiplying every other cell in matrix m by -1.

Transpose matrix m 

Multiply matrix m by (1 / (the determinant of the orignal matrix))

The result is the inverse matrix.

**Solving systems of linear equations**

x	+	y	+	z	=	6
2y + 5z =	−4
2x + 5y	−	z	=	27

This system can be represented in the following matrix form...

```
[1 1 1      [ x       [ 6 
 0 2 5   *    y    =   -4
 2 5 -1]      z ]      27 ]
```

The [x y z] matrix goes there, since, if you matrix multiply the left hand of the equation..

```
[1 1 1      [ x       [ x + y + z
 0 2 5   *    y    =    2y + 5z
 2 5 -1]      z ]       2x + 5y - 2 ]
```

which produces the same equations we had earlier


If we write

```
[1 1 1      [ x       [ 6 
 0 2 5   *    y    =   -4
 2 5 -1]      z ]      27 ]
```

as 

```
AX = B
```

A is the matrix of coefficients of the variables
X is the matrix of variables
B is the result of the equations in the system

The solution can be calculated by matrix multiplying the inverse of A with B (X = A<sup>-1</sup>B)

In our case...

```
[ x               [ -27 6 3     [ 6        [ 5 
  y   =  (1/-21) *  10 -3 5   *   -4    =    3 
  z ]               -4 -3 2 ]     27 ]       -2 ]
```

x = 5
y = 3
z = -2
