# sgd

## Run locally
```
python ./stochastic_gradient_descent.py

```

## Run as IPython notebook
```
docker run -it --rm -v ~/projects/sgd/:/projects/sgd -p 8888:8888 jupyter/scipy-notebook
```

## Questions 1-3

```
1.
	Step-size - How large the weights will change after computing the gradient.
    Number of iterations - How many iterations gradient descent will go through.
    Batch Fraction - Used to determine how many data points to use for each gradient step.

2. 
    Step-size should be relatively small in order to ensure 
    that gradient descent doesn’t diverge. 
    
    Number of iterations is a trade-off between finding a more optimal solution
    on the training set and the time to complete all iterations. 
    In some cases it might be enough to find weights that are relatively close
    to the optimal weights and not waste resources on finding the optimal weights.

    The smaller batch fraction is, the less examples are going
    to be used to compute the next gradient step. If batch fraction
    is too small, then the weights might change a lot as 
    more and more gradient steps are taken and might not always decrease the error.  
    There are no guarantees of the convergence of the algorithm when using SGD,
    but if the parameters of the algorithm are set properly,
    it should be able to find weights that are relatively close to the optimal weights. 

3.  
    LS equation - sum(residuals^2) 
    SGD equation for computing error - 1/2n * (x*weights - y)^2
    n = number of data points
    x = data set of parameters of data points
    y = output for each x
    weights = weights associated with each parameter in x
    The 2 equations are similar and resemble the same form.
```

## Questions 6-8

```
6. 
	The error doesn’t decrease monotonously with iterations,
	it oscillates around the optimal value for error.
	There is a stochastic behavior, since during iterations
	only part of the set is used to compute the next gradient step. 

7.
	Might not always converge and not have the most optimal solution. 
	During each iteration, SGD relies on the changes of each gradient 
	descent step and distributing the calcutations of the gradients 
	may not be optimal in this situation.
8. 
	Using regular gradient descent can be an alternative to SGD. 
	The benefits of using gradient descent is that it is possible
	to distribute the data, compute all the gradients, and combine the results 
	together to perform the next gradient descent step. 
	Since SGD uses only part of the data to compute the next gradient step,
	it is less efficient to distribute the data for SGD. 
	However, regular gradient descent will converge slower than SGD. 
```
