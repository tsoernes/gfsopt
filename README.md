# gfsopt
[![Documentation Status](https://readthedocs.org/projects/gfsopt/badge/?version=latest)](https://gfsopt.readthedocs.io/en/latest/?badge=latest)
[![Latest Version](https://pypip.in/version/gfsopt/badge.svg)](https://pypi.python.org/pypi/gfsopt/)

`pip3 install --user gfsopt`

Convenient scaffolding for the excellent
[Global Function Search](http://dlib.net/optimization.html#global_function_search) 
hyperparameter optimizer from the [Dlib](http://dlib.net) library. 
(See: ['A Global Optimization Algorithm Worth Using'](http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html))

Provides the following features:
* Parallel optimization: Run multiple hyperparameter searches in parallel
* Save and restore progress: Save/restore settings and optimization progress to/from file. 
* Average over multiple runs: Run a stochastic objective function using the same
parameters multiple times and report the average to Dlib's Global Function
Search. Useful in highly stochastic domains to avoid biasing the search towards
lucky runs.
# Example usage
A basic example where we maximize `obj_func` with respect to `y` over 10 runs,
with as many parallel processes as there are logical cores, and save progress to file.
```
def obj_func(x, y, pid):
    """"Function to be maximized (pid is iteration number)""""
    a = (1.5 - x + x * y)**2
    b = (2.25 - x + x * y * y)**2
    c = (2.625 - x + x * y * y * y)**2
    return -(a + b + c)
# For this example we pretend that we want to keep 'x' fixed at 0.5
# while optimizing 'y' in the range -4.5 to 4.5
pp = {'x': 0.5}  # Problem parameters
space = {'y': [-4.5, 4.5]}  # Parameters to optimize over
optimizer = GFSOptimizer(pp, space, fname="test.pkl")
# Will sample and test 'y' 10 times, then save results, progress and settings to file
optimizer.run(obj_func, n_sims=10)
```
For a more extensive example, see 
[example.py](https://github.com/tsoernes/gfsopt/blob/master/example.py).

# Installation & Requirements
Requires Python >=3.6 and the following libraries:
```
datadiff
dlib
numpy
```

To install, do:

`pip3 install --user gfsopt`

# Documentation
See [example.py](https://github.com/tsoernes/gfsopt/blob/master/example.py) for
an example and [http://gfsopt.readthedocs.io/](http://gfsopt.readthedocs.io/)
for API documentation.
