---
layout: post
katex: True
title: "Deep Learning Library From Scratch 4: Automatic Differentiation"
---


Welcome to part 4 of this series, where we will talk about automatic differentiation. 

## What is automatic differentiation?

Firstly, we need to recap on what a derivative is. 

In simple terms, a derivative of a function with respect to a variable measures how much the result of the function would change with a change in the variable. It essentially measures how sensitive the function is to a change in that variable. This is an essential part of training neural networks.

So far in our library, we have been calculating derivatives of variables by hand. However, in practice, deep learning libraries rely on **automatic differentiation**.

Automatic differentiation is the process of accurately calculating derivates of any numerical function expressed as code. 

In simpler terms, for any calculations we perform in our code, we should be able to calculate the derivates of any variables used in that calculation.

```python
...
y = 2*x + 10
y.grad(x) #what is the gradient of x???
...
```

## Forward-mode autodiff and reverse-mode autodiff

There are two popular methods of performing automatic differentiation: forward-mode and reverse-mode.

Forward-mode utilises **dual numbers** to compute derivatives.

A dual number is anything number in the form...

$$
x = a + bε
$$

where $$ ε $$ is a really small number close to 0, such that  $$ ε^2 = 0 $$

If we apply a function to a dual number as such...

$$
x = a + bε \newline
f(x) = f(a + bε) = f(a) + (f'(a) \cdot b)ε
$$

you can see we calculate both the result of $$ f(a) $$ and the gradient of $$ a $$, given by the coefficient of $$ ε $$.

Forward-mode is preferred when the input dimensions are smaller than the output dimensions of the function, however, in a deep learning setting, the input dimensions would be larger than that of the output. Reverse-mode is preferred for this situation.

In our library, we will implement reverse-mode differentiation for this reason.

Reverse-mode differentiation is a bit more difficult to implement.

As calculations are performed, a **computation graph** is built.

For example, the following diagram shows the computation graph for $$ f(x) = \frac{2x^2 + 2y}{4} $$


![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/s3iqp262sft3tk4szd21.png)

After this graph is built, the function is evaluated.

Using the function evaluation and the graph, derivatives of all variables used in the function can be calculated.

This is because each operator node would each come with a mechanism to calculate the partial derivatives of the nodes that it involves.

If we look at the bottom right node of the diagram (the $y^2$ node), the multiplier node should be able to calculate it's derivative with respect to the "y" node and the "2" node. 

Each operator node would have different mechanisms, since the way a derivative is calculated depends on the operation involved.

When using the graph to calculate derivatives, I find it easier to traverse the graph in a depth-first manner. You start at the very top node and calculate it's derivative with respect to the next node (remember, you are traversing depth-first) and record that node's gradient. Move down to that node and repeat the process. Each time you move down a level in the graph, multiply the gradient you just calculated by the gradient you calculated in the previous level (this is due to the **chain rule**). Repeat until all the nodes' gradients have been recorded.

Note: it is not necessary to calculate all the gradients in the graph. If you want to find the gradient of a single variable, you can stop once it's gradient has been calculated. However, we'd usually want to find the gradients many variables, so calculating all the gradients in the graph all at once would be much computationally cheaper, since it would only require one graph evaluation. If you wanted to find the gradients of all the variables you wanted ONLY, you would have to do an evaluation of the graph for each variable, which would turn out to be much more computationally expensive to do. 

## Differentiation rules

Here are the different differentiation rules used by each node, which are used in calculating the derivates in the computation graph.

Note: all of these will show the **partial derivative**, meaning everything that is not the variable we are finding the gradient of is treated as a constant.


In the following, think of $x$ and $y$ as nodes in the graph and $$z$$ as the result of the operation applied between these nodes.


At multiplication nodes...

$$
z = xy \newline  
\frac{dz}{dx} = y \newline
\frac{dz}{dy} = x
$$

At division nodes...

$$
z = \frac{x}{y} \newline  
\frac{dz}{dx} = \frac{1}{y} \newline
\frac{dz}{dy} = -xy^{-2}
$$

At addition nodes...

$$
z = x + y \newline  
\frac{dz}{dx} = 1 \newline  
\frac{dz}{dy} = 1
$$

At subtraction nodes...

$$
z = x - y \newline  
\frac{dz}{dx} = 1 \newline  
\frac{dz}{dy} = -1
$$

At power nodes...

$$
z = x^y \newline   
\frac{dz}{dx} = yx^{y-1} \newline  
\frac{dz}{dy} = x^y \cdot ln(x)
$$

The chain rule is then used to backpropogate all the gradients in the graph...

$$
y = f(g(x)) \newline
\frac{dy}{dx} = f'(g(x)) \cdot g'(x)
$$

However, when matrix multiplying, the chain rule get a bit different..

$$
z = x \cdot y \newline
\frac{dz}{dx} = f'(z) \otimes y^T \newline
\frac{dz}{dy} = x^T \otimes f'(z)
$$

... where $$f(z)$$ is a function that involves $$z$$, meaning $$f'(z)$$ would be the gradient calculated in the previous layer of the graph. By default (aka if z is the highest node in the graph), $$f(z) = z$$, meaning $$f'(z)$$ would be a matrix of 1s with the same shape as $$z$$

## The code

The Github repo I linked at the start contains all the code for the automatic differentiation part of the library and has updated all the neural network layers, optimisers and loss function to use automatic differentiation to calculate gradients.

To avoid this post being too long, I will show and explain the code in the next post!

Thank you for reading! 


