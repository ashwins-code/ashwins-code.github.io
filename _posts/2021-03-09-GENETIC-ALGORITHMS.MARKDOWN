---
layout: post
title: "Machine Learning Algorithms: Genetic Algorithms"
---


Hello! Welcome to this post about genetic algorithms. We will discuss how they work and implement a simple model of it in Python. Please also check my github profile [here](https://github.com/ashwins-code/)

**What is a genetic algorithm?**

A genetic algorithm is a process, inspired by natural human evolution, which aims to produce a target by "reproducing offspring" from the "fittest individuals" in a population

**How does it work?**

There are 5 main steps of genetic algorithms.

1. Initial Population
2. Determine fitness
3. Selection
4. Crossover
5. Mutation

To help explain, we will be using an example all throughout this post. 

The aim of our genetic algorithm is to produce the phrase "hello".

**Initial Population**

Every genetic algorithm needs to have an inital population, made up of individuals. This inital population is randomly generated. The size of this inital population has to be specified. In the case of our example, the initial population would just consist of random string of letters. 

Here is our initial population for our example. The population size is 6

```
asbii
jdgnf
ajdfn
fdgko
ehfgi
hrlto
```

Notice how each individual has the length 5 (same as the length of "hello"). This is because it will make the evolution process much faster, meaning that we will reach the goal faster.

**Determine fitness**

A fitness function determines how good or how close an individual is to the target. Normally, 0 is used to represent the lowest fitness and 1 is used to represent the highest score. The fitness function will change depending on the use case. In our example, the fitness function can be what fraction of letters are in the right place.

For example, "hrlto" has a score of 0.6, because it has 3 out of 5 letters which are in the right place, when compared to our target ("hello"). "ehfgi" would have a score of 0, since no letters are in the correct place. 

If we sort our intial population by their fitness, it would look like this (least fit to fittest).

```
asbii
jdgnf
ajdfn
fdgko
ehfgi
hrlto
```

**Selection**

Selection is the process where you take the best individuals in the current population to produce offspring, to create the next generation. Ideally, this next generation should improve from the last. Pairs are randomly selected from the population and will produce 2 offspring each. We can remove the weaker individuals from the population. This makes the process faster since only the fitter individuals are producing offspring. 

In our example, let's remove the 2 weakest individuals.

```
ajdfn
fdgko
ehfgi
hrlto
```


Also, we can use the concept of elitism. This means that the best individuals make it straight to the enxt generation.

Let's add the top 2 individuals in our population to the new population. So that our new population looks like this

```
ehfgi
hrlto
```

Then randomly select pairs, and remove the pairs from the population

```
ajdfn + ehfgi
hrlto + fdgko
```

The offspring of these pairs will then be added to the new population. Offspring are produced by Crossover and Mutation.


**Crossover**

Crossover is the process where the some characteristics of two parents are exchanged. A crossover point is chosen at random and each characteristic is swapped until the crossover point is reached.

Example

```
PARENTS
hrlto
fdgko
```

A crossover point is chosen
```
hr|lto
fd|gko
```

The charactersitcs until the character point are then swapped, producing two new offsrping

```
fdlto
hrgko
```

The offspring must now go through mutation before they are added to the new population

**Mutation**

In mutation, an individual is mutated by randomly changing some characteristics of it. This is done by iterating through the characterstics and deciding whether to mutate the current characteristic or not. This is done by randomly generating a number between 0 and 1 and seeing it is below the mutation rate (which you specify). If it is, you change the characteristic else leave it as it is.

For example, a mutation of "fdlto" may look like this:

```
felro
```

In the case of our example, a mutation would be changing the letter to another random letter of the alphabet.

So now, if we take the offspring from each pair produced from crossover and mutate them, we can add them to the new population so that it looks something like this:

```
ejdfl
aeaff
felro
herko
```

The steps 2 - 5 are repeated with each generation. Eventually, there would be a individual produced with a fitness of 1, meaning we have achieved our goal.


**Code**

The code below solves the problem we went through this post. The target phrase is "genetic algorithms are very cool"


```python
import random

target = "genetic algorithms are very cool"
alphabet = list("abcdefghijklmnopqrstuvwxyz ")

def create_popn(size):
    popn = []
    
    for i in range(size):
        individual = ""
        for j in range(len(target)):
            individual += random.choice(alphabet)
        popn.append(individual)

    return popn

def fitness(individual):
    correct = 0
    for ch1, ch2 in zip(individual, target):
        if ch1 == ch2:
            correct += 1

    return correct / len(target)

def crossover(p1, p2):
    crossover_point = random.randint(0, len(p1))

    offspring1 = p2[:crossover_point] + p1[crossover_point:]
    offspring2 = p1[:crossover_point] + p2[crossover_point:]
    return [offspring1, offspring2]

def mutate(individual):
    mutation_rate = 0.1
    new = ""
    for ch in individual:
        if random.random() < mutation_rate:
            new  += random.choice(alphabet)
        else:
            new += ch

    return new

def breed(p1, p2):
    return [mutate(x) for x in crossover(p1, p2)]

def evolve_popn(popn):
    popn = sorted(popn, key = lambda x : fitness(x))
    fittest_score = fitness(popn[-1])
    print (f"BEST IN GENERATION {popn[-1]}, FITNESS {fittest_score}") #Printing the best individual so we can see progress
    if fittest_score == 1:
        return 0 #break out the function since we have found the target
    popn = popn[int(len(popn) * 0.2):] #removing the worst 20% individuals
    new_popn = popn[int(len(popn) * 0.85):] #adding the top 15% individuals straight into the new population

    while len(popn) > 2:
        #selecting pairs are removing it from population
        index1 = random.choice(list(range(len(popn))))
        parent1 = popn[index1]
        popn = popn[:index1] + popn[index1+1:]
        index2 = random.choice(list(range(len(popn))))
        parent2 = popn[index2]
        popn = popn[:index2] + popn[index2+1:]

        
        #producing the offspring and adding it to the new generation
        for offspring in breed(parent1, parent2):
            new_popn.append(offspring)

    return new_popn


popn = create_popn(5000) #create an initial population of 5000

while len(popn) > 1:
    popn = evolve_popn(popn)
    if popn == 0:
        break
```

There is not enough space to show the entire output, but here are the first few lines of the output

```
BEST IN GENERATION zeadtgzeewohdut ks asjfygyvjsnpe, FITNESS 0.1875
BEST IN GENERATION umsztjf zfhdrlamys bsjlpejehmyop, FITNESS 0.21875
BEST IN GENERATION  enobjcct  xuotxaokucyos uetlool, FITNESS 0.21875
BEST IN GENERATION grbljqjeuawgrdtfxrfuyejvlrn exge, FITNESS 0.21875
BEST IN GENERATION ymsztjf zwgdrfthibpab cvyfzofjcj, FITNESS 0.25
BEST IN GENERATION  rne habozomrdthsowamhgwqrsbborz, FITNESS 0.25
BEST IN GENERATION h letpc alnmji hqjoptesbbrpkydtb, FITNESS 0.3125
BEST IN GENERATION h letpy alnmji hrigzrb vxrvvvpcd, FITNESS 0.34375
BEST IN GENERATION xvf tiy adlornrdfxdaae  vjyuqoiv, FITNESS 0.34375
BEST IN GENERATION m lutpc alnmjiwhqjoptesumrfgcokb, FITNESS 0.34375
BEST IN GENERATION h letpc alnmji hqjoptesverwufomq, FITNESS 0.40625
BEST IN GENERATION h letpc alnmji hqjoptesverwufomq, FITNESS 0.40625
```

As you can see, these individuals are far from the target. However, you can see that the fitness is progressing through each generation (meaning they are getting closer to the target)


Here is the middle of the output

```
BEST IN GENERATION geneticyalrorithvsoadefvery cool, FITNESS 0.8125
BEST IN GENERATION geneticyalrorithvsoadefvery cool, FITNESS 0.8125
BEST IN GENERATION geneticyalrorithvsoadefvery cool, FITNESS 0.8125
BEST IN GENERATION geneticyalrorithvsoadefvery cool, FITNESS 0.8125
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
BEST IN GENERATION geneticyalrorithvsoaue very cool, FITNESS 0.84375
```

Now you can see it is almost there. The best individual tends to stay the same for many generations at this stage, but a better individual will eventually come due to mutations.


```
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very ccol, FITNESS 0.96875
BEST IN GENERATION genetic algorithms are very cool, FITNESS 1.0
```

And here we achieve our target!


Although we went through many generations to achieve this, it only took about 3 seconds for us to come to it.
