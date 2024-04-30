import pickle
import sys
from functools import partial
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from typing import Callable, TypeVar

import dlib
import numpy as np
from datadiff import diff

PathOrStr = Path | str


class GFSOptimizer:
    def __init__(
        self,
        pp: dict,
        space: dict[str, tuple[int, int] | tuple[float, float]],
        solver_epsilon: float = 0.0005,
        relative_noise_magnitude: float = 0.001,
        path: PathOrStr | None = None,
        save: bool = False,
    ):
        """
        `Global Function Search (GFS) Optimizer.
        <http://dlib.net/optimization.html#global_function_search>`_

        Creates a GFS optimizer for optimizing a set of hyperparameters.
        Supports multicore parallel optimization runs and averaging over
        stochastic optimization runs. Supports saving/restoring both
        settings and progress to file.

        To create, specify settings 'pp'+'space' and/or file name 'path'.
        If 'path' is given, attempt to restore progress and settings from file.
        If restoring fails, continue with specified/default settings
        i.e. ``(pp, space, solver_epsilon, relative_noise_magnitude)``.
        If restoring succeeds, then any setting passed as argument in addition
        to the file name will be compared to the setting restored from the file,
        and you will be given the choice of which one to use.

        Parameters
        ----------
        pp: Fixed Problem Parameters.
            All fixed hyperparameters and their values for the objective
            function. E.g: ``{'beta': 0.44}``.
            Can be an empty dict if there are no fixed parameters.
        space: The parameter space for each of the hyperparameters to
            optimize over.
            Entries should be of the form:
            ``parameter: (Low_Bound, High_Bound)`` e.g:
            ``{'alpha': (0.65, 0.85), 'gamma': (1, 8)}``. If both bounds for a
            parameter are Ints, then only integers within the (inclusive) range
            will be sampled and tested.
        solver_epsilon: The accuracy to which local optima
            are determined before global exploration is resumed.
            See `Dlib <http://dlib.net/dlib/global_optimization/
            global_function_search_abstract.h.html#global_function_search>`_
            for further documentation.
        relative_noise_magnitude: (optional) Should be increased for
            highly stochastic objective functions. Deterministic and continuous
            functions can use a value of 0. See `Dlib
            <http://dlib.net/dlib/global_optimization/upper_bound_function_abstract.h.html
            #upper_bound_function>`_
            for further documentation.
        path: File path or name for restoring and/or saving results,
            progress and settings.
        save: Save settings and progress periodically,
            on user quit (CTRL-C), and on completion.
        """

        # Verify inputs
        if path:
            # Save or restore
            if not Path(path).exists():
                if pp is None or space is None:
                    raise ValueError(
                        "Either pp or space is not given. Attempted "
                        "to load from file, but path does not exist."
                    )
        else:
            if pp is None or space is None:
                raise ValueError("Either pp or space is not given.")
            if save:
                raise ValueError(
                    "If you want to save you must specify a file name `path`."
                )
        eps = solver_epsilon
        noise_mag = relative_noise_magnitude

        # Parse arguments into a format suitable for Dlib
        params = []  # parameters to optimize over
        # Whether a parameter should be optimized in integer steps or not
        is_int = []
        # Lower and upper bounds within which to optimize parameters
        lo_bounds, hi_bounds = [], []
        if space is not None:
            for parm, conf in space.items():
                params.append(parm)
                lo, hi = conf
                is_int.append(isinstance(lo, int) and isinstance(hi, int))
                lo_bounds.append(lo)
                hi_bounds.append(hi)
        old_evals = []
        if path is not None:
            try:
                # Load progress and settings from file, then compare each
                # restored setting with settings specified by args (if any)
                old_raw_spec, _old_spec, old_evals, info, prev_best = _load(path)
                saved_params = info["params"]
                print(
                    f"Restored {len(old_evals)} trials, prev best: "
                    f"{prev_best[0]}@{list(zip(saved_params, prev_best[1:]))}"
                )
                if params and params != saved_params:
                    # Switching params being optimized over would throw off Dlib.
                    # Must use restored params instead of specified parameters.
                    print(
                        f"Warning: Saved params {saved_params} differ from currently specified "
                        f"{params}. Using saved."
                    )
                params = saved_params
                if is_int:
                    raw_spec = _cmp_and_choose(
                        "bounds", old_raw_spec, (is_int, lo_bounds, hi_bounds)
                    )
                else:
                    raw_spec = old_raw_spec
                is_int, lo_bounds, hi_bounds = raw_spec
                if len(params) != len(is_int):
                    raise ValueError(
                        f"Params {params} and spec {raw_spec} are of different length"
                    )
                eps = _cmp_and_choose("solver_epsilon", info["solver_epsilon"], eps)
                noise_mag = _cmp_and_choose(
                    "relative_noise_magnitude",
                    info["relative_noise_magnitude"],
                    noise_mag,
                )
                _, pp = _compare_pps(info["pp"], pp)
            except FileNotFoundError:
                # Create a new file. Dlib does this for us
                pass

        # Initialize the GFS optimizer
        spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
        if old_evals:
            optimizer = dlib.global_function_search(
                [spec],
                initial_function_evals=[old_evals],
            )
        else:
            optimizer = dlib.global_function_search([spec])
        optimizer.set_solver_epsilon(solver_epsilon)
        optimizer.set_relative_noise_magnitude(relative_noise_magnitude)

        self.pp, self.params, self.optimizer, self.spec = pp, params, optimizer, spec
        self.eps, self.noise_mag = eps, noise_mag
        self.is_int = is_int
        self.path, self.save = path, save

    def run(
        self,
        obj_func: Callable[..., float | int],
        n_concurrent: int | None = None,
        n_avg: int = 1,
        n_sims: int = 1000,
        save_iter: int = 30,
    ):
        """
        Run optimization.

        Parameters
        ----------
        obj_func: Function to maximize.
            The function must take as arguments every parameter specified in
            the union of  'pp' and 'space', in addition to 'pid',
            and return the result as float.
            'pid', an integer, specifies simulation run number.
            If you want to minimize instead,
            simply negate the result in the objective function before returning it.
        n_concurrent: Number of concurrent processes.
            If 'None' (the default), then use as all logical cores.
        n_avg: Number of runs to average results over. Useful to counteract noise
            when optimizing stochastic functions.
        n_sims: Number of times to sample and test params.
        save_iter: How often to save progress.
        """
        if n_concurrent is None:
            n_concurrent = cpu_count()
        if n_concurrent <= 0:
            raise ValueError(f"{n_concurrent=} must be > 0")
        if n_avg <= 0:
            raise ValueError(f"{n_avg=} must be > 0")
        if n_concurrent % n_avg != 0:
            # TODO This is a pretty hefty restriction. Why was this necessary?
            raise ValueError(
                f"n_avg ({n_avg}) must divide n_concurrent ({n_concurrent}) evenly"
            )
        if n_sims < n_concurrent:
            raise ValueError(
                "Must have more simulations to run in total than in parallel"
            )
        n_step = n_concurrent // n_avg

        # Becomes populated with results as simulations finishes
        result_queue = Queue()
        simproc = partial(
            _dlib_proc, obj_func, self.pp, self.params, self.is_int, result_queue
        )
        # Becomes populated with evaluation objects to be set later
        evals = [None] * n_sims
        # Becomes populates with losses. When n_avg losses for a particular
        # set of params are ready, their mean is set for the corresponding eval.
        results = [[] for _ in range(n_sims)]

        def save_evals():
            """Store results of finished evals to file; print best eval"""
            finished_evals = self.optimizer.get_function_evaluations()[1][0]
            _save(
                self.spec,
                finished_evals,
                self.params,
                self.eps,
                self.noise_mag,
                self.pp,
                self.path,
            )
            print(f"Saving {len(finished_evals)} trials to {self.path}.")
            print_best()

        def print_best():
            """Print best run thus far"""
            best_eval = self.optimizer.get_best_function_eval()
            prms = list(zip(self.params, list(best_eval[0])))
            res = best_eval[1]
            print(f"Best eval so far: {res}@{prms}")

        def spawn_evals(i):
            """Spawn a new optimizer process"""
            eeval = self.optimizer.get_next_x()
            evals[i] = eeval  # Store eval object to be set with result later
            vals = list(eeval.x)
            # print(f"T{i} Testing {self.params}: {vals}")
            for _ in range(n_avg):
                Process(target=simproc, args=(i, vals)).start()

        def store_result():
            """Block until a result is ready, then store it and report it to dlib"""
            try:
                # Blocks until a result is ready
                i, result = result_queue.get()
            except KeyboardInterrupt:
                # Handle 'ctrl-c'
                if self.save:
                    inp = ""
                    while inp not in ["Y", "N"]:
                        inp = input("Premature exit. Save? Y/N: ").upper()
                    if inp == "Y":
                        save_evals()
                else:
                    print_best()
                sys.exit(0)
            else:
                if result is not None:
                    results[i].append(result)
                    # Wait until 'n_avg' results are finished for the same set
                    # of params before reporting (mean) result to GFS
                    if len(results[i]) == n_avg:
                        evals[i].set(np.mean(results[i]))
                if i > 0 and i % save_iter == 0 and len(results[i]) == n_avg:
                    if self.save:
                        save_evals()
                    else:
                        print_best()

        print(
            f"Optimizing for {n_sims} sims with {n_concurrent} concurrent processes,"
            f" for each set of parameters taking the average of {n_avg} runs,"
            f" optimizing over parameters {self.params} with `solver_epsilon={self.eps}`"
            f" and `relative_noise_magnitude={self.noise_mag}`."
        )
        # Spawn initial processes
        for i in range(n_step):
            spawn_evals(i)
        # When a thread returns a result, start a new sim
        for i in range(n_step, n_sims):
            for _ in range(n_avg):
                store_result()
            spawn_evals(i)
        # Get remaining results
        for _ in range(n_step):
            for _ in range(n_avg):
                store_result()
        if self.save:
            save_evals()
        else:
            print_best()
        print("Finished.")


X = TypeVar("X")


def _cmp_and_choose(what: str, saved: X, specified: X) -> X:
    """
    Compare and choose between settings loaded from file (saved)
    or settings given as paremeter on GFSOptimizer initilization.
    """
    chosen = saved
    if specified and saved != specified:
        print(
            f"Saved {what}={saved} differ from currently specified {what}={specified}"
        )
        inp = ""
        while inp not in ["N", "Y"]:
            inp = input(f"Use saved {what} (Y) or specified {what} (N)?: ").upper()
        if inp == "N":
            chosen = specified
    return chosen


def _compare_pps(old_pp: dict, new_pp: dict) -> tuple[bool, dict]:
    """
    Given two sets of problem params, compare them and if they differ, ask
    which one to use and return it (use_old_pp, pp)
    """
    pp = old_pp
    use_old_pp = True
    if new_pp and old_pp != new_pp:
        pp_diff = diff(old_pp, new_pp)
        if "dt" in old_pp:
            print(f"Found old problem params from file stored at {old_pp['dt']}")
        print(f"Diff('a': old, from file. 'b': specified, from args):\n{pp_diff}")
        ans = ""
        while ans not in ["Y", "N"]:
            ans = input("Use old pp (Y) instead of specified (N)?: ").upper()
        if ans == "N":
            use_old_pp = False
            pp = new_pp
    return (use_old_pp, pp)


def _dlib_proc(obj_func, pp, space_params, is_int, result_queue, i, space_vals):
    """
    Add/overwrite problem params with params given from dlib,
    then run objective function and put the result in the given queue
    """
    for j, key in enumerate(space_params):
        pp[key] = int(space_vals[j]) if is_int[j] else space_vals[j]
    result = obj_func(**pp, pid=i)
    result_queue.put((i, result))


def _save(spec, evals, params, solver_epsilon, relative_noise_magnitude, pp, path):
    """
    Save progress and settings to a pickle file with file name 'path'.
    See documentation for 'load' for parameter specification.
    """
    raw_spec = (list(spec.is_integer_variable), list(spec.lower), list(spec.upper))
    raw_results = np.zeros((len(evals), len(evals[0].x) + 1))
    info = {
        "params": params,
        "solver_epsilon": solver_epsilon,
        "relative_noise_magnitude": relative_noise_magnitude,
        "pp": pp,
    }
    for i, eeval in enumerate(evals):
        raw_results[i][0] = eeval.y
        raw_results[i][1:] = list(eeval.x)
    with open(path, "wb") as f:
        pickle.dump((raw_spec, raw_results, info), f)


def _load_raw(path):
    with open(path, "rb") as f:
        raw_spec, raw_results, info = pickle.load(f)
    return raw_spec, raw_results, info


def _load(path):
    """
    Load a pickle file containing
    (spec, results, info) where
      results: np.array of shape [N, M+1] where
        N is number of trials
        M is number of hyperparameters
        results[:, 0] is result/loss
        results[:, 1:] is [param1, param2, ...]
      spec: (is_integer, lower, upper)
        where each element is list of length M
      info: dict with keys
        params, solver_epsilon, relative_noise_magnitude, pp

    Assumes only 1 function is optimized over

    Returns
    (dlib.function_spec, [dlib.function_eval], dict, prev_best)
      where prev_best: np.array[result, param1, param2, ...]
    """
    raw_spec, raw_results, info = _load_raw(path)
    is_integer, lo_bounds, hi_bounds = raw_spec
    spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_integer)
    evals = []
    prev_best = raw_results[np.argmax(raw_results, axis=0)[0]]
    for raw_result in raw_results:
        x = list(raw_result[1:])
        result = dlib.function_evaluation(x=x, y=raw_result[0])
        evals.append(result)
    return raw_spec, spec, evals, info, prev_best


def print_best(path, n=1, minimum=False):
    """
    Load results from file specified by file name 'path'
    and print the 'n' best results, where best means maximum
    by default and minimum if 'minimum' is specified.

    :param str path: File name
    :param int n: (optional) Number of results to print
    :param bool minimum: (optional) If lower result is better
    """
    raw_spec, raw_results, info = _load_raw(path)
    is_integer, lo_bounds, hi_bounds = raw_spec
    rs = raw_results[raw_results[:, 0].argsort()]
    if minimum:
        losses = rs[-n:, 0][::-1]
        parms = rs[-n:, 1:][::-1]

    res_str = ""
    for i in range(len(losses)):
        pa = [f"{p}:{v}" for p, v in zip(info["params"], parms[i])]
        # pa = list(zip(parms[i]
        lo = f"{losses[i]} "
        res_str += lo + " ".join(pa) + "\n"

    bound_vals = [f"{lo}<>{hi}" for lo, hi in zip(raw_spec[1], raw_spec[2])]
    bounds = [f"{prm}: {bnd}" for prm, bnd in zip(info["params"], bound_vals)]
    bounds_str = "\n".join(bounds)

    print(f"Loaded {path}. Settings:")
    print(*info.items(), sep="\n")
    print(f"Bounds (param: lo_bound<>hi_bound):\n{bounds_str}")
    print(f"Found {len(raw_results)} results. Top {n}:\n{res_str}")
