from main import GFSOptimizer, print_best


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
    """
    Objective function to be _maximized_ by Dlib.
    """
    res = beale(x, y)
    print(f"Iter: {pid}\t x:{x}, y:{y}, result:{res}")
    # Since Dlib maximizes, but we want to find the minimum,
    # we negate the result before passing it to the Dlib optimizer.
    return -res


# For this example we pretend that we want to keep 'x' fixed at 0.5
# while optimizing 'y' in the range -4.5 to 4.5
pp = {'x': 0.5}
space = {'y': [False, -4.5, 4.5]}
fname = "beale.pkl"
optimizer = GFSOptimizer(pp, space, fname=fname)
optimizer.run(obj_func, n_sims=100)

# Can also load previously used settings ('pp' and 'space' from file)
# optimizer = GFSOptimizer(fname=fname)
# optimizer.run(obj_func, n_sims=100)

print("\n\n")
print_best(fname, 5, minimum=True)
