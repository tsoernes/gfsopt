# dlib-optimize

convenient scaffolding 

Convenience wrapper around the 
[Global Function Search](http://dlib.net/optimization.html#global_function_search) 
hyperparameter optimizer from the [Dlib](http://dlib.net) library.
Provides the following features:
* Parallel optimization: Run multiple hyperparameter searches in parallel
* Save and restore progress: Save and restore Dlib's internal progress, hyperparameters being optimized over, other hyperparamaters, and progress to/from file.
* Average over multiple runs: Run a stochastic simulation using the same parameters multiple times and 
report the average to Dlib's Global Function Search. Useful in highly stochastic domains to avoid biasing the search towards lucky runs.

# Example usage

# Installation & Requirements
datadiff
dlib

# API
## Creating optimizers
Create optimizer from scratch
```
# Hyperparameters being optimized over.
# Optimize non-int parameter 'beta' in range 0.5 to 0.8
space = {'beta': (False, 0.5, 0.8)}  
# Other hyperparamaters needed by objective function
pp = {'alpha': 1}  
opt = optimizer_from_file(pp, space)
```

Restore progress and settings from file
```
opt = optimizer_from_file("progress.pkl")
```

Restore progress and settings from file. 
Check if saved hyperparameters differ from currently specified ones,
and if they do, ask by terminal input which ones to use
```
pp = {'alpha': 1}
space = {'beta': (False, 0.5, 0.8)}
opt = optimizer_from_file("progress.pkl", pp, space)
```

# TODO
Don't want to force restore or force saving

# Cases
* Specify pp, space. Don't restore, don't save.
* Specify pp, space, fname. Don't restore, but save.
* Specify pp, space, fname. Attempt restore. If restoring fails, create new. Else, compare. Save.
* Specify fname. Attempt restore. If restoring fails, must fail loudly. Save ??

Defining the API is a bit awkward because:
* Don't want to require pp+space because restoring from fname should suffice
* Don't want to require fname because pp+space should suffice, 
    and saving should not be a requirement
* Want to allow both pp+space AND fname to be specified on loading, 
   in order to compare saved vs specified conf
