import pickle
import sys
from functools import partial
from multiprocessing import Process, Queue, cpu_count
import os.path

import dlib
import numpy as np
from datadiff import diff


class DlibRunner():
    def __init__(self,
                 pp=None,
                 space=None,
                 solver_epsilon=None,
                 relative_noise_magnitude=None,
                 fname=None,
                 save=True):
        """
        If 'fname' is given, attempt to restore progress and settings from file.
        If restoring fails, continue with specified settings
        i.e. (pp, space, solver_epsilon, relative_noise_magnitude).
        If restoring succeeds, then any settings passed as argument in addition
        to the file name will be compared to the settings restored from the file
        and you will be given the option of whether to use
        argument settings or file settings.

        pp: dict with all hyperparameters for the objective function,
            including those not being optimized over. Can, but does not need to,
            include hyperparameters being optimized over.
            If a hyperparameter is specified both in 'pp' and in 'space',
            its value in 'pp' will be overridden.

        space: dict with hyperparameters to optimize over.
            entries should be of the form:
            parameter: (IsInteger, Low_Bound, High_Bound)
            e.g.:
            'alpha': (False, 0.65, 0.85)

        solver_epsilon: float, see Dlib documentation. Default: 0.0005

        relative_noise_magnitude: float, see Dlib documentation. Default: 0.001

        fname: file name for restoring and/or saving progress

        save: bool, Whether to save progress periodically,
            on user-quit (ctrl-c), and on completion.
        """
        if fname is None:
            assert pp is not None and space is not None, \
                "You must specify either file name or pp + space"
            eps = solver_epsilon if solver_epsilon is None else 0.0005
            if relative_noise_magnitude is None:
                noise_mag = 0.001
            else:
                noise_mag = relative_noise_magnitude
            assert not save, "If you want to save you must specify a file name"
        else:
            if not os.path.isfile(fname):
                if not os.path.isfile(fname + ".pkl"):
                    raise FileNotFoundError(fname)
                else:
                    fname = fname + ".pkl"

        params, is_int, lo_bounds, hi_bounds = [], [], [], []
        if space is not None:
            for parm, conf in space.items():
                params.append(parm)
                is_int.append(conf[0])
                lo_bounds.append(conf[1])
                hi_bounds.append(conf[2])
        if fname is not None:
            # Load progress and settings from file.
            old_raw_spec, old_spec, old_evals, info, prev_best = dlib_load(fname)
            saved_params = info['params']
            print(f"Restored {len(old_evals)} trials, prev best: "
                  f"{prev_best[0]}@{list(zip(saved_params, prev_best[1:]))}")
            if params and params != saved_params:
                # Switching params being optimized over would throw off Dlib.
                # Must use restore params from specified
                print(f"Saved params {saved_params} differ from currently specified "
                      f"{params}. Using saved.")
            params = saved_params
            raw_spec = _cmp_and_choose('bounds', old_raw_spec,
                                       (is_int, lo_bounds, hi_bounds))
            spec = dlib.function_spec(
                bound1=raw_spec[1], bound2=raw_spec[2], is_integer=raw_spec[0])
            eps = _cmp_and_choose('solver_epsilon', info['solver_epsilon'], eps)
            noise_mag = _cmp_and_choose('relative_noise_magnitude',
                                        info['relative_noise_magnitude'], noise_mag)
            _, pp = compare_pps(info['pp'], pp)
            optimizer = dlib.global_function_search(
                [spec],
                initial_function_evals=[old_evals],
                relative_noise_magnitude=noise_mag)
        else:
            spec = dlib.function_spec(
                bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
            optimizer = dlib.global_function_search(spec)
            optimizer.set_relative_noise_magnitude(noise_mag)
        optimizer.set_solver_epsilon(eps)
        self.pp, self.params, self.optimizer, self.spec = pp, params, optimizer, spec
        self.fname, self.save = fname, save

    def run(self, obj_func, n_concurrent=None, n_avg=1, n_sims=1000, save_iter=30):
        """
        obj_func: function to maximize. Must take as argument every parameter specified in
            both 'pp' and 'space', in addition to 'pid', and return a float.
        n_concurrent: int, Number of concurrent procs.
            If 'None' or not specified, then use as all threads
        n_avg: int, Number of runs to average over
        n_sims: int, Number of times to sample and test params
        save_iter: int, How often to save progress
        """
        assert type(n_concurrent) is int
        assert type(n_avg) is int
        assert n_concurrent > 0
        assert n_avg > 0
        if n_concurrent is None:
            n_concurrent = cpu_count()
        assert n_concurrent % n_avg == 0, \
            f"n_avg ({n_avg}) must divide n_concurrent ({n_concurrent}) evenly"
        n_step = n_concurrent // n_avg
        eps, noise_mag = self.solver_epsilon, self.relative_noise_magnitude

        # Becomes populated with results as simulations finished
        result_queue = Queue()
        simproc = partial(dlib_proc, self.stratclass, self.pp, self.params, result_queue)
        # Becomes populated with evaluation objects to be set later
        evals = [None] * n_sims
        # Becomes populates with losses. When n_avg losses for a particular
        # set of params are ready, their mean is set for the corresponding eval.
        results = [[] for _ in range(n_sims)]

        # TODO
        # - Only save if 'save'
        # - Make sure results are printed if not 'save'

        def save_evals():
            """Store results of finished evals to file; print best eval"""
            finished_evals = self.optimizer.get_function_evaluations()[1][0]
            dlib_save(self.spec, finished_evals, self.params, eps, noise_mag, self.pp,
                      self.fname)
            best_eval = self.optimizer.get_best_function_eval()
            prms = list(zip(self.params, list(best_eval[0])))
            print(f"Saving {len(finished_evals)} trials to {self.fname}."
                  f"Best eval so far: {best_eval[1]}@{prms}")

        def spawn_evals(i):
            """Spawn a new sim process"""
            eeval = self.optimizer.get_next_x()
            evals[i] = eeval  # Store eval object to be set with result later
            vals = list(eeval.x)
            print(f"T{i} Testing {self.params}: {vals}")
            for _ in range(n_avg):
                Process(target=simproc, args=(i, vals)).start()

        def store_result():
            """Block until a result is ready, then store it and report it to dlib"""
            try:
                # Blocks until a result is ready
                i, result = result_queue.get()
            except KeyboardInterrupt:
                inp = ""
                while inp not in ["Y", "N"]:
                    inp = input("Premature exit. Save? Y/N: ").upper()
                if inp == "Y":
                    save_evals()
                sys.exit(0)
            else:
                if result is not None:
                    results[i].append(result)
                    if len(results[i]) == n_avg:
                        evals[i].set(np.mean(results[i]))
                if i > 0 and i % save_iter == 0 and len(results[i]) == n_avg:
                    save_evals()

        print(f"Dlib hopt for {n_sims} sims with {n_concurrent} procs"
              f" taking the average of {n_avg} runs"
              f" on params {self.space} and s.eps {eps}")
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
        save_evals()


def _cmp_and_choose(what, saved, specified):
    chosen = saved
    if specified and saved != specified:
        print(f"Saved {what} {saved} differ from currently specified {specified}")
        inp = ""
        while inp not in ['N', 'Y']:
            inp = input(f"Use saved {what} (Y) instead of specified (N)?: ").upper()
        if inp == "N":
            chosen = specified
    return chosen


def dlib_proc(obj_func, pp, space_params, result_queue, i, space_vals):
    # Add/overwrite problem params with params given from dlib
    for j, key in enumerate(space_params):
        pp[key] = space_vals[j]
    result = obj_func(pp=pp, pid=i)
    result_queue.put((i, result))


def compare_pps(old_pp, new_pp):
    """
    Given two sets of problem params, compare them and if they differ, ask
    which one to use and return it (use_old_pp, pp)
    """
    pp_diff = diff(old_pp, new_pp)
    pp = new_pp
    use_old_pp = False
    if old_pp != new_pp:
        if 'dt' in old_pp:
            print(f"Found old problem params from file stored at {old_pp['dt']}")
        print(f"Diff('a': old, from file. 'b': specified, from args):\n{pp_diff}")
        ans = ''
        while ans not in ['y', 'n']:
            ans = input("Use old pp (Y) instead of specified (N)?: ").lower()
        if ans == 'y':
            use_old_pp = True
            pp = old_pp
    return (use_old_pp, pp)


def dlib_save(spec, evals, params, solver_epsilon, relative_noise_magnitude, pp, fname):
    """
    Save progress and settings to 'fname'.
    See documentation for 'load' for parameter specification.
    """
    raw_spec = (list(spec.is_integer_variable), list(spec.lower), list(spec.upper))
    raw_results = np.zeros((len(evals), len(evals[0].x) + 1))
    info = {
        'params': params,
        'solver_epsilon': solver_epsilon,
        'relative_noise_magnitude': relative_noise_magnitude,
        'pp': pp,
    }
    for i, eeval in enumerate(evals):
        raw_results[i][0] = eeval.y
        raw_results[i][1:] = list(eeval.x)
    with open(fname, "wb") as f:
        pickle.dump((raw_spec, raw_results, info), f)


def dlib_load_raw(fname):
    with open(fname, "rb") as f:
        raw_spec, raw_results, info = pickle.load(f)
    return raw_spec, raw_results, info


def dlib_load(fname):
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
    raw_spec, raw_results, info = dlib_load_raw(fname)
    is_integer, lo_bounds, hi_bounds = raw_spec
    spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_integer)
    evals = []
    prev_best = raw_results[np.argmax(raw_results, axis=0)[0]]
    for raw_result in raw_results:
        x = list(raw_result[1:])
        result = dlib.function_evaluation(x=x, y=raw_result[0])
        evals.append(result)
    return raw_spec, spec, evals, info, prev_best


def dlib_best(fname, n=1, minimum=False):
    """
    Load results from file specified by 'fname',
    and print 'n' best results, where best means maximum
    by default and minimum if 'minimum' is specified.
    """
    raw_spec, raw_results, info = dlib_load_raw(fname)
    is_integer, lo_bounds, hi_bounds = raw_spec
    rs = raw_results[raw_results[:, 0].argsort()]
    if minimum:
        # Losses are stored as negative so negate.
        losses = rs[-n:, 0][::-1]
        parms = rs[-n:, 1:][::-1]

    res_str = ""
    for i in range(len(losses)):
        pa = [f"--{p} {v}" for p, v in zip(info['params'], parms[i])]
        # pa = list(zip(parms[i]
        lo = f"{losses[i]:.4f} "
        res_str += lo + " ".join(pa) + "\n"

    bound_vals = [f"{lo}<>{hi}" for lo, hi in zip(raw_spec[1], raw_spec[2])]
    bounds = [f"{prm}: {bnd}" for prm, bnd in zip(info['params'], bound_vals)]
    bounds_str = ", ".join(bounds)

    print(*info.items(), sep="\n")
    print(f"{len(raw_results)} results\n{bounds_str}\n{res_str}")
