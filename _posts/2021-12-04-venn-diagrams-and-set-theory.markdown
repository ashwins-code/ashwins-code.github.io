---
layout: post
title: "Maths for Computer Science : Venn Diagrams and Set Theory"
---


### Sets

A set in mathematics is simply collection of elements and all these elements have a common property with each other.

Sets are written with curly brackets, with the elements written between those brackets, separated with commas.

E.g

```
{2, 4, 6, ...}
```

Capitals letters are usually used to represent sets...

```
A = {0, 2, 4, 6} 
```

The order of elements in a set does not matter, so two sets, who have the exact same members, are equal...

```
A = {1, 2, 3}
B = {3, 2, 1}

A is equal to B
```

To show if an element is a member of a set, we use the symbol **∈**

**∉** is used to show if an element is not a member of a set

```
A = {1, 2, 3}

1 ∈ A
4 ∉ A
```

### Unions

The union of two sets joins two sets together.

The union of two sets is represented with the symbol **∪**

```
A = {1, 3 ,5}
B = {2, 4, 6}

A ∪ B = {1, 2, 3, 4, 5, 6}
```

A ∪ B would create a set which contains members in A **or** B.

### Intersections

The intersection of two sets is a set that contains elements that belong to both of those two sets.

The intersection of two sets is represetned with the symbol **∩**

```
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

A ∩ B = {3, 4}
```

A ∩ B represens the set of elements in A **and** B.

### Venn Diagrams

Venn Diagrams are used to show the relationship between different sets.

A circle is used to represent a set, with the elements of that set written inside that circle...

```
A = {1, 2, 3, 4, 5, 6}
B = {4, 5, 6, 7, 8, 9}
````

![Alt Text](/assets/venn-diagram-1.png) 

The area where the two circles overlap represents the set **A ∩ B**

**A ∪ B** can be found simply just be reading the elements that are in circle A and circle B. 

Venn Diagrams can also be used to show the relation between more than 2 sets.

```
A = {1, 2, 3}
B = {2, 3, 4, 7}
C = {3, 4, 5}
```

![Alt Text](/assets/venn-diagram-2.png) 

By looking at the circle overlaps we can see...

```
A ∩ B = { 2, 3 }
A ∩ C = { 3 }
B ∩ C = { 3, 4 }
```

and reading off all elements in the circles would give

```
A ∪ B ∪ C = {1, 2, 3, 4, 5, 7}
```

To get A ∪ B, we read the values in the circles A and B

```
A ∪ B = {1, 2, 3, 4, 7}
```

To get A ∪ C, we read values in circles A and C

```
A ∪ C = {1, 2, 3, 4, 5}
```

To get B ∪ C, we read the values in circles B and C

```
B ∪ C = {2, 3, 7, 4, 5}
```
