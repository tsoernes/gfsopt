# dlib-optimize

Convenience wrapper around [Global Function Search](http://dlib.net/optimization.html#global_function_search) hyperparameter optimizer from the [Dlib](http://dlib.net) library.
Provides the following features:
* Parallel optimization: Run multiple hyperparameter searches in parallel
* Save and restore progress: Save and restore Dlib's internal progress, hyperparameters being optimized over, other hyperparamaters, and progress
* Average over multiple runs: Run a stochastic simulation using the same parameters multiple times and 
report the average to Dlib's Global Function Search. Useful in highly stochastic domains to avoid biasing the search towards lucky runs.

# Example usage

# Installation & Requirements
datadiff
dlib

# API
