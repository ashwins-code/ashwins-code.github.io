---
layout: post
title: "Make a Calculator with JavaScript: Part 1, Tokenizer"
---

Hi! Welcome to this post where we will be creating a calculator from scratch in JavaScript. 

This is part 1 where we will look at tokenizing. Part 2 is where we will look at evaluating and part 3 we will look at creating the UI.

#### What is tokenizing?

Tokenizing means taking in a string and separating the *tokens* found in that string. In the case of a calculation, the different tokens would be the numbers and the operators.

Our tokenizer should be able to take in a calculation, which is a string, and return an array of the tokens within that string

For example...

```
calculation = "(10+12)/5"

getTokens(calculation) => ["(","10", "+", "12", ")", "/", "5"]
```

Tokenizing is important since it allows our program to understand what our user has entered and it makes life easier when we come to evaluating our expression

#### Approach

First, we would need a tokens array to hold all the tokens we find

```
tokens = []
```

The key to this problem is to have a *currentToken* variable, which would start as an empty string. 

When we iterate through our given calculation, the current character will be appended to *currentToken*. We will keep appending to this variable until it becomes a valid token, at which point we would append it to the tokens list, make it an empty string and the process continues. In other words, *currentToken* is being treated as a buffer. 

Let's use an example to see how this appoach works

```
Example Calculation: (10.5 - 12)/-5
```

```
Iterate through the calculation

ITERATION 1: currentToken = (
currentToken is a valid token. Append it to tokens list and make currentToken empty
Tokens: (

ITERATION 2: currentToken = 1

ITERATION 3: currentToken = 10

ITERATION 4: currentToken = 10.

ITERATION 5: currentToken = 10.5
currentToken is a valid token. Append it to tokens list and make currentToken empty
Tokens: (,10.5

ITERATION 6: currentToken = -
currentToken is a valid token. Append it to tokens list and make currentToken empty
Tokens: (,10.5,-

ITERATION 7: currentToken = 1

ITERATION 8: currentToken = 12
currentToken is a valid token. Append it to tokens list and make currentToken empty
Tokens: (,10.5,-,12

ITERATION 9: currentToken = )
currentToken is a valid token. Append it to tokens list and make currentToken empty
Tokens: (,10.5,-,12,)

ITERATION 10: currentToken = /
currentToken is a valid token. Append it to tokens list and make currentToken empty
Tokens: (,10.5,-,12,),/

ITERATION 11: currentToken = -

ITERATION 12: currentToken = -5
currentToken is a valid token. Append it to tokens list and make currentToken empty
Tokens: (,10.5,-,12,),/,-5
```

As you might have noticed, we did not add *currentToken* at iteration 11, despite it being a valid token!

This is because we wanted to add the negative number instead (-5) in this case.

To make sure we get negative numbers as a token, we only add the "-" token by itself if the last token we added to the tokens list was a number. This way, we can tell the difference between a subtraction of two numbers and a negative number by itself. 

Also, to make sure we capture the full number, everytime *currentToken* is a number, we look one character ahead and see if this can be part of a number. If it is, we do not add *currentToken* to the tokens list. If it is not, we add *currentToken* to the tokens list and make *currentToken* empty

#### Code

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
            }else if (!isNaN(tokens[tokens.length - 1])) {
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

console.log(getTokens("(10.5-12)/-5"))

// => ["(", 10.5 ,"-", 12, ")" , "/" ,-5]
```
