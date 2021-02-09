---
layout: post
title: "Make a Calculator with JavaScript: Part 2, Evaluator"
---

Hello! Welcome to part 2 of "Make a Calculator with JavaScript". In this post, we talk about evaluating expressions.

#### Infix to Postfix

The key aspect of evaluating expressions is converting infix expressions to postfix expressions.

An infix expression is an expression which is in the form of [operand] [operator] [operand] e.g 5 + 10

A postfix expression is an expression  which is in the form of [operand] [operand] [operator] e.g 5 10 +

#### Why do we need to convert to postfix?

Postfix is a format that makes it easier for us to evaluate our expression compared to if it was in an infix format. This is because a postfix expression has the order of operations to perform from left to right, which is not always the case with an infix expression. 

#### Infix to Postfix algorithm

From part 1, we have converted our expression into a list of tokens. Therefore, we just need to rearrange this list of tokens so that it is in a postfix form.

Example

```
infixToPostfix(['(',10,'+',5,'*',4,')']) => [10,5,4,'*','+']
```

```
RANK OF OPERATORS

^: 3
*: 2
/: 2
+: 1
-: 1
```

```
1. Have an output list and a stack
2. Scan the infix expression from left to right
3. If the scanned character is a number, push it to the output list
  - If it is not a number
    - If the scanned character is "("
        -Push it to output list
    - If the scanned character is an operator
        - While the last element on the stack has a greater or equal rank to the scanned operator
            -Pop the last element and push it to the output list
        - Then push the scanned operator
    - If the scanned character is ")"
        - Keep popping the stack until a "(" is encountered. Pop the "(" too.
```

#### Infix to Postfix Code

```js

function infixToPostfix(tokens) {
    var stack = []
    var result = []
    var rank = {
        "^": 3,
        "*": 2,
        "/": 2,
        "+": 1,
        "-": 1
    }
    var ops = Object.keys(rank)
    for (var t of tokens) {
        if (t == "(") {
            stack.push(t)
        }else if (ops.indexOf(t) > -1) {
            while (rank[stack[stack.length - 1]] >= rank[t]) {
                result.push(stack.pop())
            }

            stack.push(t)
        }else if (t == ")") {
            while (stack.length != 0 && stack[stack.length - 1] != "(") {
                result.push(stack.pop())
            }
            stack.pop()
        }else {
            result.push(t)
        }
    }

    result = [...result, ...stack]

    return result
}

```

#### Evaluating our Expression


Evaluating an expression is quite simple with what we have done now. Here is the algorithm


```
1. Surround the expression with "(" and ")" and then tokenize it
2. Convert the tokens list to postifx form
3. Have an empty stack
4. Scan the postfix expression
5. If the scanned element is a number, push it to the stack
   - If it is an operator
    - Pop the last element of the stack (this is the right hand side of the operation)
    - Pop the last element again (this is the left hand side of the operation)
    - Perform the required operation on these numbers and push the result on to the stack
6. By the end of scanning through the expression, there should be one element left in the stack, which is the result of the expression
```


#### Evaluation code


```js
function evaluate(expr) {
    var tokens = getTokens("("+expr+")")
    var postfix = infixToPostfix(tokens)
    var stack = []

    for (var token of postfix) {
        if (!isNaN(token)) {
            stack.push(token)
        }else {
            var rhs = stack.pop()
            var lhs = stack.pop()

            if (token == "+") {
                stack.push(lhs + rhs)
            }else if (token == "-") {
                stack.push(lhs - rhs)
            }else if (token == "*") {
                stack.push(lhs * rhs)
            }else if (token == "^") {
                stack.push(lhs ** rhs)
            }else if (token == "/") {
                stack.push(lhs / rhs)
            }
        }
    }

    return stack[0]
}
```


#### All the code (including part 1)


```js
function getTokens(calculation) {
    var tokens = []
    var currentToken = ""
    var operators = ["+", "-", "/", "*", "^", "(", ")"]

    for (var i = 0; i < calculation.length; i++) {
        currentToken += calculation[i]

        if (operators.indexOf(currentToken) > -1) {
            if (currentToken != "-") {
                tokens.push(currentToken)
                currentToken = ""
            }else if (!isNaN(tokens[tokens.length - 1]) || tokens[tokens.length - 1] == ")") {
                tokens.push(currentToken)
                currentToken = ""
            }
        }else if (!isNaN(currentToken) && isNaN(currentToken + calculation[i+1])) {
            tokens.push(Number(currentToken))
            currentToken = ""
        }
    }

    return tokens
}

function infixToPostfix(tokens) {
    var stack = []
    var result = []
    var rank = {
        "^": 3,
        "*": 2,
        "/": 2,
        "+": 1,
        "-": 1
    }
    var ops = Object.keys(rank)
    for (var t of tokens) {
        if (t == "(") {
            stack.push(t)
        }else if (ops.indexOf(t) > -1) {
            while (rank[stack[stack.length - 1]] >= rank[t]) {
                result.push(stack.pop())
            }

            stack.push(t)
        }else if (t == ")") {
            while (stack.length != 0 && stack[stack.length - 1] != "(") {
                result.push(stack.pop())
            }
            stack.pop()
        }else {
            result.push(t)
        }
    }

    result = [...result, ...stack]

    return result
}

function evaluate(expr) {
    var tokens = getTokens("("+expr+")")
    var postfix = infixToPostfix(tokens)
    var stack = []

    for (var token of postfix) {
        if (!isNaN(token)) {
            stack.push(token)
        }else {
            var rhs = stack.pop()
            var lhs = stack.pop()

            if (token == "+") {
                stack.push(lhs + rhs)
            }else if (token == "-") {
                stack.push(lhs - rhs)
            }else if (token == "*") {
                stack.push(lhs * rhs)
            }else if (token == "^") {
                stack.push(lhs ** rhs)
            }else if (token == "/") {
                stack.push(lhs / rhs)
            }
        }
    }

    return stack[0]
}

console.log(evaluate("(10+5)*4")) // => 60
console.log(evaluate("(11/5)*((8-2)+(6^2))")) // => 92.4

```
