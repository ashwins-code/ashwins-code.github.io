---
layout: post
title: "Leetcode problem 216: Combination Sum III"
---

Hello! Welcome to this post where we will solve the medium level difficulty, Combination Sum III.

**The Problem**

Find all valid combinations of k numbers that sum up to n such that the following conditions are true:

Only numbers 1 through 9 are used.
Each number is used at most once.
Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.

  
```
Example 1:

Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.
Example 2:

Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]
Explanation:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
There are no other valid combinations.
Example 3:

Input: k = 4, n = 1
Output: []
Explanation: There are no valid combinations.
Using 4 different numbers in the range [1,9], the smallest sum we can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.
Example 4:

Input: k = 3, n = 2
Output: []
Explanation: There are no valid combinations.
Example 5:

Input: k = 9, n = 45
Output: [[1,2,3,4,5,6,7,8,9]]
Explanation:
1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 45
There are no other valid combinations.
```

**The Solution**

The key to this problem is to have a recursive, helper function that takes in a target, a combination list (this list is used to hold a possible valid combination) and another list, which would hold the remaining numbers of 1 to 9 that are available.

This function would go through the list of numbers, and subtract each number from the target, appending the current number to the current combination list. It then removes the number from the numbers list.

It would then call on itself, passing in the new target, new combination list and the new numbers list.

If the passed target is 0 and the length of the combination list is equal to k, we know we have a valid combination, so that is appended to our results.

This function would initial be called passing n as the target and an empty list. 

For example...

```
k = 2, n = 6
```

Let's call our recursive function "build", with parameters "target", "combination" and "options"

```
build(n, [], [1,2,3,4,5,6,7,8,9])
```

It would first subtract 1 from the target (n) and append it to the combination

```
target = 5
combination = [1]
options = [2,3,4,5,6,7,8,9]
```

It would now call itself with these new parameters

```
build(target, list, options)
```

It would now subtract 2 from the target and append it to the combination
```
target = 3
combination = [1, 2]
options = [3,4,5,6,7,8,9]
```

It would now call itself with these new parameters

```
build(target, list, options)
```

It would now subtract 3 from the target and append it to the combination
```
target = 0
combination = [1, 2, 3]
options = [4,5,6,7,8,9]
```

Our target is now 0 and our list length is equal to k, meaning our list is a valid combination. It is added to our results list

```
results = [[1, 2, 3]]
```

This is the process just to find one combination. However, this recursive process is repeated for each of the number from 1 to 9. It is quite hard to understand, but looking at the code itself would make it  easier to understand. 

**The Code**

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        options = list(range(1, 10))
        result = []
        
        def build(target, combination, options):
            if len(combination) > k:
                return 0
            
            if target == 0 and len(combination) == k:
                result.append(combination)
                
            for i in range(len(options)):
                num = options[i]
                
                if target - num >= 0:
                    build(target - num, combination+[num], options[i+1:])

        build(n, [], options)
                    
        return result
```