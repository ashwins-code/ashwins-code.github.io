---
layout: post
title: "Analytic Geometry"
---

Hello and welcome to this post where we will talk about Analytic Geometry.


**What is Analytic Geometry?**

Analytic geometry covers geometric objects on a coordinate plane. This includes finding the lengths of vectors, distance between vectors and finding the angle between vectors.

**Norms**

A norm is a function that determines the length of a vector. There are different norms, meaning that different norms may produce different lengths for the same vector.

One norm is the Manhattan norm.

The Manhattan norm takes a vector and returns the sum of all the absolute values of the vector.

For example

```
Vector: [4 -3 5 -1]
```

The length of this vector (if applied to the Manhattan Norm) would be 13 (4+3+5+1)

Another norm is called the Euclidean norm.

This returns the straight line distance from the origin and the vector.

It is calculated by square rooting the sum of all the squared values of the vector/

For example

```
Vector: [2 4 -3]
```

The length of this vector, if calculated using the Euclidean norm, would be 25 (2^2 + 4^2 + (-3)^2)


**Inner Products**

An inner product is a way to multiply vectors together, such that the result of the multiplication is a scalar value. 

**Dot Product**

The dot product is a type of inner product.

It is calculated by taking the sum of the multiplications of corresponding values between 2 vectors.

For example

```
Vector A [1 2 3]
Vector B [4 5 6]
```

The dot product of A and B would be (1 * 4)+(2 * 5)+(3 * 6)=32

**Angles**

Inner products also allow us to find the angle between vectors.

The angle between a vector A and vector B can be calculated as follows

angle = cos<sup>-1</sup> ((inner product of A and B) / (norm of A * norm of B))

**Orthogonality**

2 vectors are orthogonal if an inner product of the 2 vectors result to be 0.

Geometrically, orthogonal vectors are at a right angle to each other. 

**Uses of Analytic Geometry**

Concepts in Analytic Geometry can be applied in recommender systems. Data can be encoded into vectors and to see how similar 2 pieces of data are can be calculated by finding the distance between the vector encodings. 

 