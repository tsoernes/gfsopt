from gfsopt import GFSOptimizer, print_best


def beale(x, y):
    """
    Beale's function (see https://en.wikipedia.org/wiki/Test_functions_for_optimization).
    Has a global _minimum_ of 0 at x=3, y=0.5.
    """
    a = (1.5 - x + x * y)**2
    b = (2.25 - x + x * y * y)**2
    c = (2.625 - x + x * y * y * y)**2
    return a + b + c


def obj_func(x, y, pid):
    """ Objective function to be _maximized_ by GFS. """
    res = beale(x, y)
    print(f"Iter: {pid}\t x:{x}, y:{y}, result:{res}")
    # Since Dlib maximizes, but we want to find the minimum,
    # we negate the result before passing it to the Dlib optimizer.
    return -res


# For this example, we pretend that we want to keep 'x' fixed at 0.5
# while optimizing 'y' in the range -4.5 to 4.5
space = {'y': [-4.5, 4.5]}
pp = {'x': 0.5}
fname = "beale.pkl"

# Create an optimizer
optimizer = GFSOptimizer(pp, space, fname=fname, save=True)
# Will run 10 simulations with as many in parallel as there are logical cores,
# then save settings and results to file.
optimizer.run(obj_func, n_sims=10)
# >>> Optimizing for 10 sims with 4 procs, for each set of params taking the average
# of 1 runs, optimizing over params ['y'] with solver_eps 0.0005 and noise mag 0.001
# >>> Iter: 0	 x:0.5, y:0.0, result:8.578125
# >>> Iter: 1	 x:0.5, y:-2.4005459633675557, result:44.4497091305862
# >>> Iter: 2	 x:0.5, y:2.36385282253673, result:101.60970694773587
# >>> Iter: 3	 x:0.5, y:-3.1824628882375263, result:242.53160521028394
# >>> Iter: 4	 x:0.5, y:-1.9637829908593951, result:16.29058881746099
# >>> Iter: 5	 x:0.5, y:-0.5090459682881638, result:8.328166942194017
# >>> Iter: 6	 x:0.5, y:-0.5446913668245081, result:8.311937990090515
# >>> Iter: 7	 x:0.5, y:4.497612103471647, result:2418.5041507192686
# >>> Iter: 8	 x:0.5, y:0.9992095021679255, result:14.192165352650964
# >>> Iter: 9	 x:0.5, y:3.430101827963232, result:563.0817236020747
# >>> Saving 10 trials to beale.pkl.
# >>> Best eval so far: -8.311937990090515@[('y', -0.5446913668245081)]

# Can also load previously used settings ('pp' and 'space') from file.
# Will restore progress from last optimization run.
optimizer = GFSOptimizer(fname=fname)
# Run 10 additional simulations
optimizer.run(obj_func, n_sims=10)
# >>> Restored 10 trials, prev best: -8.311937990090515@[('y', -0.5446913668245081)]
# >>> Optimizing for 10 sims with 4 procs, for each set of params taking the average
# of 1 runs, optimizing over params ['y'] with solver_eps 0.0005 and noise mag 0.001
# >>> Iter: 0	 x:0.5, y:-1.0621481653077909, result:7.903116185791083
# >>> Iter: 1	 x:0.5, y:-4.497270171028074, result:2021.9072963359108
# >>> Iter: 2	 x:0.5, y:1.6571726062085474, result:32.46178151358101
# >>> Iter: 3	 x:0.5, y:-3.839844544668535, result:769.6162895854932
# >>> Iter: 4	 x:0.5, y:-2.0970617622743566, result:21.77629250825386
# >>> Iter: 5	 x:0.5, y:0.4999174550684051, result:9.862887911306508
# >>> Iter: 6	 x:0.5, y:-1.5111180112494056, result:8.581663227025247
# >>> Iter: 7	 x:0.5, y:2.7763709300884853, result:201.6020038606197
# >>> Iter: 8	 x:0.5, y:-1.5796049637910736, result:9.053459949824337
# >>> Iter: 9	 x:0.5, y:1.3223786435728302, result:20.41307664793507
# >>> Saving 20 trials to beale.pkl.
# >>> Best eval so far: -7.903116185791083@[('y', -1.0621481653077909)]

# Let's pretend we only want to test integer values for 'y',
space = {'y': [-4, 4]}
# while keeping 'x' as before.
# As we create an optimizer and pass both settings and filename for a
# file with saved settings, GFSOptimizer will check if the settings match.
optimizer = GFSOptimizer(pp, space, fname=fname)
# As they do not ('y' now only takes integer values; and the range is different),
# you'll get the option of using the settings found in the file:
# >>> Restored 20 trials, prev best: -7.903116185791083@[('y', -1.0621481653077909)]
# >>> Saved bounds ([False], [-4.5], [4.5]) differ from currently specified ([True], [-4], [4])
# >>> Use saved bounds (Y) instead of specified (N)?: N
# >>> Optimizing for 10 sims with 4 procs, for each set of params taking the average
# of 1 runs, optimizing over params ['y'] with solver_eps 0.0005 and noise mag 0.001
# >>> Iter: 0	 x:0.5, y:2, result:55.578125
# >>> Iter: 1	 x:0.5, y:3, result:289.453125
# >>> Iter: 2	 x:0.5, y:-1, result:7.953125
# >>> Iter: 3	 x:0.5, y:-3, result:168.703125
# >>> Iter: 4	 x:0.5, y:-2, result:17.578125
# >>> Iter: 5	 x:0.5, y:3, result:289.453125
# >>> Iter: 6	 x:0.5, y:1, result:14.203125
# >>> Iter: 7	 x:0.5, y:-3, result:168.703125
# >>> Iter: 8	 x:0.5, y:-2, result:17.578125
# >>> Iter: 9	 x:0.5, y:1, result:14.203125
# >>> Saving 30 trials to beale.pkl.
# >>> Best eval so far: -7.953125@[('y', -1.0)]
# >>> Finished.

optimizer.run(obj_func, n_sims=10)

# Load pickle file and print 5 best results
print_best(fname, 5, minimum=True)
# >>> Loaded beale.pkl. Settings:
# >>> ('params', ['y'])
# >>> ('solver_epsilon', 0.0005)
# >>> ('relative_noise_magnitude', 0.001)
# >>> ('pp', {'x': 0.5})
# >>> Bounds (param: lo_bound<>hi_bound):
# >>> y: -4.0<>4.0
# >>> Found 30 results. Top 5:
# >>> -7.903116185791083 y:-1.0621481653077909
# >>> -7.953125 y:-1.0
# >>> -8.311937990090515 y:-0.5446913668245081
# >>> -8.328166942194017 y:-0.5090459682881638
# >>> -8.578125 y:0.0
