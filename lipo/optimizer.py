"""
The optimization module
"""
# import logging
import math
import time
from typing import Callable, Dict, List, Tuple, Union
from multiprocessing import Process, Queue, Manager, cpu_count, Lock, Pool
from functools import partial
import numpy as np

import sys
import signal

# from queue import Queue
# import queue

# import dill
# from multiprocessing import reduction

# reduction.ForkingPickler = dill.Pickler

import dlib

# logger = logging.getLogger(__name__)

print('Hello from my LIPO fork')

class EvaluationCandidate:
    def __init__(self, candidate, arg_names, categories, log_args, maximize, is_integer):
        self.candidate = candidate
        self.arg_names = arg_names
        self.maximize = maximize
        self.log_args = log_args
        self.categories = categories
        self.is_integer = is_integer

    @property
    def x(self):
        x = {}
        for name, val in zip(self.arg_names, self.candidate.x):
            if self.is_integer[name]:
                val = int(val)
            if name in self.categories:
                x[name] = self.categories[name][val]
            elif name in self.log_args:
                x[name] = math.exp(val)
            else:
                x[name] = val
        return x

    def set(self, y):
        if not self.maximize:
            y = -y
        self.candidate.set(y)


class GlobalOptimizer:
    """
    Global optimizer that uses an efficient derivative-free method to optimize.

    See
    `LIPO algorithm implementation <http://dlib.net/python/index.html#dlib.find_max_global>`_. A good explanation of
    how it works, can be found here: `Here <http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html>`_
    """

    def __init__(
        self,
        function: Union[Callable, None] = None,
        lower_bounds: Dict[str, Union[float, int]] = {},
        upper_bounds: Dict[str, Union[float, int]] = {},
        limit_bounds: Dict[str, List[float]] = {},
        categories: Dict[str, List[str]] = {},
        log_args: Union[str, List[str]] = "auto",
        flexible_bounds: Dict[str, List[bool]] = {},
        flexible_bound_threshold: float = -1.0,
        evaluations: List[Tuple[Dict[str, Union[float, int, str]], float]] = [],
        maximize: bool = True,
        epsilon=0.0,
        random_state=None,
        random_search_probability=0.02,
        num_random_samples=5000,
        manager=None,
        shared_attribute=None,
    ):
        """
        Init optimizer

        Args:
            function (callable): function to optimize
            lower_bounds (Dict[str]): lower bounds of optimization, integer arguments are automatically inferred
            upper_bounds (Dict[str]): upper bounds of optimization, integer arguments are automatically inferred
            log_args (Union[str, List[str]): list of arguments to treat as if in log space, if "auto", then
                a variable is optimized in log space if

                - The lower bound on the variable is > 0
                - The ratio of the upper bound to lower bound is > 1000
                - The variable is not an integer variable
            flexible_bounds (Dict[str, List[bool]]): dictionary of parameters and list of booleans indicating
                if parameters are deemed flexible or not. By default all parameters are deemed flexible but only
                if `flexible_bound_threshold > 0`.
            flexible_bound_threshold (float): if to enlarge bounds if optimum is top or bottom
                ``flexible_bound_threshold`` quantile
            evaluations List[Tuple[Dict[str], float]]: list of tuples of x and y values
            maximize (bool): if to maximize or minimize (default ``True``)
            epsilon (float): accuracy below which exploration will be priorities vs exploitation; default = 0
            random_state (Union[None, int]): random state
        """
        self.function = function
        self.epsilon = epsilon
        self.random_search_probability = random_search_probability
        self.num_random_samples = num_random_samples
        self.random_state = random_state

        self.manager = Manager()
        self.shared_attribute = self.manager.list()

        # check bounds
        assert len(lower_bounds) == len(upper_bounds), "Number of upper and lower bounds should be the same"
        for name in lower_bounds.keys():
            is_lower_integer = isinstance(lower_bounds[name], int)
            is_upper_integer = isinstance(upper_bounds[name], int)
            assert (is_lower_integer and is_upper_integer) or (
                not is_lower_integer and not is_upper_integer
            ), f"Argument {name} must be either integer or not integer"
            assert (
                lower_bounds[name] < upper_bounds[name]
            ), f"Lower bound should be smaller than upper bound for argument {name}"

        self.categories = categories
        # obtain all the keys that have a defined bound, either is a category or not
        self.arg_names = list(upper_bounds.keys())
        for key in self.categories:
            # if the category has a bound, we dont need to check
            if key in self.arg_names:
                continue
            # if the category has no bound, add it to the list
            self.arg_names.append(key)

        # set bounds
        # default limit [0.01, 0.99]
        # self.limit_bounds = {name: limit_bounds.get(name, [0.01, 0.99]) for name in self.arg_names}
        self.limit_bounds = dict()
        for name in self.arg_names:
            # if name is in categories, we dont need to check
            if name in self.categories:
                continue
            self.limit_bounds[name] = limit_bounds.get(name, [0.01, 0.99])

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        for name, cats in self.categories.items():
            # if name is in lower bounds, we dont need to check
            if name in lower_bounds:
                continue
            # if the category has no bound, set it to the length of the category. Eg: this is the case of lookbackwards in our code. This way we do not need to specify the bounds in main().
            self.lower_bounds[name] = 0
            self.upper_bounds[name] = len(cats) - 1

        # infer if variable is integer
        self.is_integer = {name: isinstance(self.lower_bounds[name], int) for name in self.arg_names}

        # set arguments in log space
        if isinstance(log_args, str) and log_args == "auto":
            self.log_args = []
            for name in self.arg_names:
                if (
                    not self.is_integer[name]
                    and self.lower_bounds[name] > 0
                    and self.upper_bounds[name] / self.lower_bounds[name] > 1e3
                ):
                    self.log_args.append(name)
        else:
            self.log_args = log_args
        # transform bounds
        for name in self.log_args:
            assert name not in self.categories, f"Log-space is not defined for categories such as {name}"
            assert not self.is_integer[name], f"Log-space is not defined for integer variables such as {name}"
            assert self.lower_bounds[name] > 0, f"Log-space for {name} is only defined for positive lower bounds"
            self.lower_bounds[name] = math.log(self.lower_bounds[name])
            self.upper_bounds[name] = math.log(self.upper_bounds[name])

        # check log args
        for name in self.log_args:
            assert not self.is_integer[name], f"Integer or categorical arguments such as {name} cannot be in log space"

        self.saved_evaluations = evaluations

        # convert initial evaluations
        self.init_evaluations = []
        # x: arg dict
        # y: list of pattern ids
        # z: Discrimination Ration
        # th: Threshold
        for x, _, z, _ in evaluations:
            e = {}
            for name, val in x.items():
                if name in self.categories:
                    #TODO: If we have save evaluations for a specific patient, it is possible that the patient is not in the categories list. In this case, we add it to the list.
                    e[name] = self.categories[name].index(val)
                elif name in self.log_args:
                    e[name] = math.log(val)
                else:
                    e[name] = val
            self.init_evaluations.append((e, z))

        # if to maximize
        self.maximize = maximize

        # check bound threshold
        assert flexible_bound_threshold < 0.5, "Quantile for bound flexibility has to be below 0.5"
        self.flexible_bound_threshold = flexible_bound_threshold
        self.flexible_bounds = {name: flexible_bounds.get(name, [True, True]) for name in self.arg_names}

        # initialize search object
        self._init_search()

    def _init_search(self):
        function_spec = dlib.function_spec(
            [self.lower_bounds[name] for name in self.arg_names],
            [self.upper_bounds[name] for name in self.arg_names],
            [self.is_integer[name] for name in self.arg_names],
        )
        self.search = dlib.global_function_search(
            functions=[function_spec],
            initial_function_evals=[
                [dlib.function_evaluation([x[0][name] for name in self.arg_names], x[1]) for x in self._raw_evaluations]
            ],
            relative_noise_magnitude=0.001,
        )
        self.search.set_solver_epsilon(self.epsilon)
        self.search.set_pure_random_search_probability(self.random_search_probability)
        self.search.set_monte_carlo_upper_bound_sample_num(self.num_random_samples)
        if self.random_state is not None:
            self.search.set_seed(self.random_state)

        # print all self.serach parameters
        # print("Search parameters:")
        # print(f"  epsilon: {self.epsilon}")
        # print(f"  random_search_probability: {self.random_search_probability}")
        # print(f"  num_random_samples: {self.num_random_samples}")
        # print(f"  random_state: {self.random_state}")

    def get_candidate(self):
        """
        get candidate for evaluation

        Returns:
            EvaluationCandidate: candidate has property `x` for candidate kwargs and method `set` to
                inform the optimizer of the value
        """
        if self.flexible_bound_threshold >= 0:  # if to flexibilize bounds

            start = time.time()

            if len(self.search.get_function_evaluations()[1][0]) > 1 / (
                max(self.flexible_bound_threshold, 0.05)
            ):  # ensure sufficient evaluations have happened -> not more than 20
                reinit = False
                # check for optima close to bounds
                optimum_args = self.optimum[0]
                for name in self.arg_names:

                    lower = self.lower_bounds[name]
                    upper = self.upper_bounds[name]
                    span = upper - lower

                    if name in self.log_args:
                        val = math.log(optimum_args[name])
                    else:
                        val = optimum_args[name]

                    if name in self.categories:
                        # It's val a index or a value
                        # print(f'get_candidate category: {name} = {val} <- {self.categories[name].index(val)}')
                        # continue

                        min_limit = 0
                        max_limit = len(self.categories[name]) - 1

                        # optimum arg value so far
                        val = self.categories[name].index(val)

                        # be sure growth is greater than 0
                        growth = max(int(max_limit*self.flexible_bound_threshold), 1)

                        # redefine lower bound
                        if (val - lower) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][0]:
                            # center value
                            # proposed_val = val - (upper - val)
                            # fixed increment
                            proposed_val = lower - growth

                            # print(f'get_candidate lower: {name} = {val} ; growth = {growth} ; proposed_val = {proposed_val}')

                            # continue

                            if proposed_val < self.lower_bounds[name]:
                                # Don't accept lower arguments than min_limit
                                if proposed_val <= min_limit:
                                    proposed_val = min_limit
                                    # no longer flexible
                                    self.flexible_bounds[name][0] = False
                                self.lower_bounds[name] = proposed_val
                                # restart search
                                reinit = True

                        # redefine upper bound
                        elif (upper - val) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][1]:
                            # center value
                            # proposed_val = val + (val - lower)
                            # fixed increment
                            proposed_val = upper + growth

                            # print(f'get_candidate upper: {name} = {val} ; growth = {growth} ; proposed_val = {proposed_val}')

                            # continue

                            if proposed_val > self.upper_bounds[name]:
                                # Don't accept arguments greater than 1.0
                                if max_limit <= proposed_val:
                                    proposed_val = max_limit
                                    # no longer flexible
                                    self.flexible_bounds[name][1] = False
                                self.upper_bounds[name] = proposed_val
                                # restart search
                                reinit = True
                    else:
                        # redefine lower bound
                        if (val - lower) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][0]:
                            # center value
                            proposed_val = val - (upper - val)
                            # fixed increment
                            # proposed_val = lower - 0.1
                            # limit change in log space
                            if name in self.log_args:
                                proposed_val = max(self.upper_bounds[name] - 2, proposed_val, -15)

                            if proposed_val < self.lower_bounds[name]:
                                # Don't accept lower arguments than min_limit
                                if proposed_val <= self.limit_bounds[name][0]:
                                    proposed_val = self.limit_bounds[name][0]
                                    # no longer flexible
                                    self.flexible_bounds[name][0] = False
                                self.lower_bounds[name] = proposed_val
                                # restart search
                                reinit = True

                        # redefine upper bound
                        elif (upper - val) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][1]:
                            # center value
                            proposed_val = val + (val - lower)
                            # fixed increment
                            # proposed_val = upper + 0.1
                            # limit log space redefinition
                            if name in self.log_args:
                                proposed_val = min(self.upper_bounds[name] + 2, proposed_val, 15)

                            if proposed_val > self.upper_bounds[name]:
                                # Don't accept arguments greater than 1.0
                                if self.limit_bounds[name][1] <= proposed_val:
                                    proposed_val = self.limit_bounds[name][1]
                                    # no longer flexible
                                    self.flexible_bounds[name][1] = False
                                self.upper_bounds[name] = proposed_val
                                # restart search
                                reinit = True

                        if self.is_integer[name]:
                            self.lower_bounds[name] = int(self.lower_bounds[name])
                            self.upper_bounds[name] = int(self.upper_bounds[name])

                if reinit:  # reinitialize optimization with new bounds
                    # logger.debug(f"resetting bounds to {self.lower_bounds} to {self.upper_bounds}")
                    self._init_search()
                    print('\t\tOPT REINIT TIME:', time.time()-start)

        return EvaluationCandidate(
            candidate=self.search.get_next_x(),
            arg_names=self.arg_names,
            categories=self.categories,
            maximize=self.maximize,
            log_args=self.log_args,
            is_integer=self.is_integer,
        )

    def get_candidate_par(self, lock):
        """
        get candidate for evaluation

        Returns:
            EvaluationCandidate: candidate has property `x` for candidate kwargs and method `set` to
                inform the optimizer of the value
        """

        # adquire lock
        lock.acquire()

        if self.flexible_bound_threshold >= 0:  # if to flexibilize bounds

            start = time.time()

            if len(self.search.get_function_evaluations()[1][0]) > 1 / (
                max(self.flexible_bound_threshold, 0.05)
            ):  # ensure sufficient evaluations have happened -> not more than 20
                reinit = False
                # check for optima close to bounds
                optimum_args = self.optimum[0]
                for name in self.arg_names:

                    lower = self.lower_bounds[name]
                    upper = self.upper_bounds[name]
                    span = upper - lower

                    if name in self.log_args:
                        val = math.log(optimum_args[name])
                    else:
                        val = optimum_args[name]

                    if name in self.categories:
                        # It's val a index or a value
                        # print(f'get_candidate category: {name} = {val} <- {self.categories[name].index(val)}')
                        # continue

                        min_limit = 0
                        max_limit = len(self.categories[name]) - 1

                        # optimum arg value so far
                        val = self.categories[name].index(val)

                        # be sure growth is greater than 0
                        growth = max(int(max_limit*self.flexible_bound_threshold), 1)

                        # redefine lower bound
                        if (val - lower) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][0]:
                            # center value
                            # proposed_val = val - (upper - val)
                            # fixed increment
                            proposed_val = lower - growth

                            # print(f'get_candidate lower: {name} = {val} ; growth = {growth} ; proposed_val = {proposed_val}')

                            # continue

                            if proposed_val < self.lower_bounds[name]:
                                # Don't accept lower arguments than min_limit
                                if proposed_val <= min_limit:
                                    proposed_val = min_limit
                                    # no longer flexible
                                    self.flexible_bounds[name][0] = False
                                self.lower_bounds[name] = proposed_val
                                # restart search
                                reinit = True

                        # redefine upper bound
                        elif (upper - val) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][1]:
                            # center value
                            # proposed_val = val + (val - lower)
                            # fixed increment
                            proposed_val = upper + growth

                            # print(f'get_candidate upper: {name} = {val} ; growth = {growth} ; proposed_val = {proposed_val}')

                            # continue

                            if proposed_val > self.upper_bounds[name]:
                                # Don't accept arguments greater than 1.0
                                if max_limit <= proposed_val:
                                    proposed_val = max_limit
                                    # no longer flexible
                                    self.flexible_bounds[name][1] = False
                                self.upper_bounds[name] = proposed_val
                                # restart search
                                reinit = True
                    else:
                        # redefine lower bound
                        if (val - lower) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][0]:
                            # center value
                            proposed_val = val - (upper - val)
                            # fixed increment
                            # proposed_val = lower - 0.1
                            # limit change in log space
                            if name in self.log_args:
                                proposed_val = max(self.upper_bounds[name] - 2, proposed_val, -15)

                            if proposed_val < self.lower_bounds[name]:
                                # Don't accept lower arguments than min_limit
                                if proposed_val <= self.limit_bounds[name][0]:
                                    proposed_val = self.limit_bounds[name][0]
                                    # no longer flexible
                                    self.flexible_bounds[name][0] = False
                                self.lower_bounds[name] = proposed_val
                                # restart search
                                reinit = True

                        # redefine upper bound
                        elif (upper - val) / span <= self.flexible_bound_threshold and self.flexible_bounds[name][1]:
                            # center value
                            proposed_val = val + (val - lower)
                            # fixed increment
                            # proposed_val = upper + 0.1
                            # limit log space redefinition
                            if name in self.log_args:
                                proposed_val = min(self.upper_bounds[name] + 2, proposed_val, 15)

                            if proposed_val > self.upper_bounds[name]:
                                # Don't accept arguments greater than 1.0
                                if self.limit_bounds[name][1] <= proposed_val:
                                    proposed_val = self.limit_bounds[name][1]
                                    # no longer flexible
                                    self.flexible_bounds[name][1] = False
                                self.upper_bounds[name] = proposed_val
                                # restart search
                                reinit = True

                        if self.is_integer[name]:
                            self.lower_bounds[name] = int(self.lower_bounds[name])
                            self.upper_bounds[name] = int(self.upper_bounds[name])

                if reinit:  # reinitialize optimization with new bounds
                    # logger.debug(f"resetting bounds to {self.lower_bounds} to {self.upper_bounds}")
                    self._init_search()
                    print('\t\tOPT REINIT TIME:', time.time()-start)

        # release lock
        lock.release()

        return EvaluationCandidate(
            candidate=self.search.get_next_x(),
            arg_names=self.arg_names,
            categories=self.categories,
            maximize=self.maximize,
            log_args=self.log_args,
            is_integer=self.is_integer,
        )

    @property
    def _raw_evaluations(self):
        if hasattr(self, "search"):
            evals = [
                ({name: val for name, val in zip(self.arg_names, e.x)}, e.y)
                for e in self.search.get_function_evaluations()[1][0]
            ]
        else:
            evals = self.init_evaluations
            # print(f'\n\nLength init_evaluations:{len(evals)}\n')
            # list_evaluations = [([x[0][name] for name in self.arg_names], x[1]) for x in evals]
            # print(f'\nlist_evaluations: {list_evaluations[:5]}\n')
        return evals

    @property
    def evaluations(self):
        """
        evaluations (as initialized and carried out)

        Returns:
            List[Tuple[Dict[str], float]]]: list of x and y value pairs
        """
        # convert log space and categories
        # converted_evals = []
        # for x, y in self._raw_evaluations:
        #     e = {}
        #     for name, val in x.items():
        #         if self.is_integer[name]:
        #             val = int(val)
        #         if name in self.categories:
        #             e[name] = self.categories[name][val]
        #         elif name in self.log_args:
        #             e[name] = math.exp(val)
        #         else:
        #             e[name] = val
        #     converted_evals.append((e, y))
        # import numpy as np
        
        # print('\n\n\nEqual saved evaluations:', np.all(np.asarray(converted_evals) == np.asarray(self.saved_evaluations)))
        # print(converted_evals[0], self.saved_evaluations[0])
        # print(len(self.saved_evaluations), len(converted_evals))
        return self.saved_evaluations

    @property
    def optimum(self):
        """
        Current optimum

        Returns:
            Tuple[Dict[str], float, float]: tuple of optimal x, corresponding y and idx or optimal evaluation
        """
        x, y, idx = self.search.get_best_function_eval()
        new_x = {}
        for name, val in zip(self.arg_names, x):
            if self.is_integer[name]:
                val = int(val)
            if name in self.categories:
                new_x[name] = self.categories[name][val]
            elif name in self.log_args:
                new_x[name] = math.exp(val)
            else:
                new_x[name] = val

        return new_x, y, idx

    def run_parallel(self, num_function_calls: int = 1, nprocs=None):
        """
        run optimization in parallel

        Args:
            num_function_calls (int): number of function calls
            nprocs (int): number of processes to use
        """
        if nprocs is None:
            nprocs = cpu_count()
        
        assert nprocs > 0, "nprocs must be positive"

        assert nprocs < num_function_calls, "Must have more evaluations than processes"

        # start_opt = time.time()

        # we need to save the candidates because the Queue will not be able to pickle them
        candidates = np.full(num_function_calls, None, dtype=object)

        # idea from https://github.com/tsoernes/gfsopt/blob/master/gfsopt/gfsopt.py#L321
        result_queue = Queue()

        def spawn_process(pid):
            # start_t = time.time()
            candidate = self.get_candidate()
            # if pid%100 == 99: # print every 100 candidates
            #     print(f'Candidate {pid} time: {time.time()-start_t :.2f} s; Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')
            # store candidate to be set with result later
            candidates[pid] = candidate
            # spawn process
            Process(target=_dlib_proc, args=(self.function, result_queue, pid, candidate.x)).start()

        def store_result():
            # get result
            pid, idxs, y, th = result_queue.get(block=True, timeout=None)
            # save evaluation
            self.saved_evaluations.append((candidates[pid].x, idxs, y, th))
            # update search
            candidates[pid].set(y)
            # print the last nprocs results executed
            # if num_function_calls-nprocs <= len(self.saved_evaluations):
            #     print(f'Candidate {pid} time: Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')

        # initialize
        for i in range(nprocs):
            spawn_process(i)
        
        # run
        for i in range(nprocs, num_function_calls):
            store_result()
            spawn_process(i)
        
        # finish
        for _ in range(nprocs):
            store_result()

        # assert len(self.saved_evaluations) == num_function_calls, "Number of evaluations does not match"

        return


    def run_parallel2(self, num_function_calls: int = 1, nprocs=None):
        """
        run optimization in parallel

        Args:
            num_function_calls (int): number of function calls
            nprocs (int): number of processes to use
        """
        if nprocs is None:
            nprocs = cpu_count()
        
        assert nprocs > 0, "nprocs must be positive"

        assert nprocs < num_function_calls, "Must have more evaluations than processes"

        # start_opt = time.time()

        # we need to save the candidates because the Queue will not be able to pickle them
        candidates = np.full(num_function_calls, None, dtype=object)

        task_queue = Queue()
        result_queue = Queue()

        def worker():
            while True:
                # get task
                pid, x = task_queue.get(block=True, timeout=None)
                # check if we are done
                if pid is None:
                    return
                # spawn process
                idxs,y,th = self.function(**x)
                # save evaluation
                result_queue.put((pid, idxs, y, th))

        def spawn_process(pid):
            # start_t = time.time()
            candidate = self.get_candidate()
            # if pid%100 == 99: # print every 100 candidates
            #     print(f'Candidate {pid} time: {time.time()-start_t :.2f} s; Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')
            # store candidate to be set with result later
            candidates[pid] = candidate
            # spawn process
            task_queue.put((pid, candidate.x))

        def store_result():
            # get result
            pid, idxs, y, th = result_queue.get(block=True, timeout=None)
            # save evaluation
            self.saved_evaluations.append((candidates[pid].x, idxs, y, th))
            # update search
            candidates[pid].set(y)
            # print the last nprocs results executed
            # if num_function_calls-nprocs <= len(self.saved_evaluations):
            #     print(f'Candidate {pid} time: Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')

        # initialize
        workers = [Process(target=worker) for _ in range(nprocs)]
        for i in range(nprocs):
            # spawn process
            spawn_process(i)
            # start worker
            workers[i].start()
        
        # run
        for i in range(nprocs, num_function_calls):
            # add task
            spawn_process(i)
            # store result
            store_result()
        
        # finish
        for _ in range(nprocs):
            # store result
            store_result()
            # add task to stop worker
            task_queue.put((None, None))
        
        for p in workers:
            p.join()

        # assert len(self.saved_evaluations) == num_function_calls, "Number of evaluations does not match"

        return


    def run_parallel3(self, num_function_calls: int = 1, nprocs=None):
        """
        Run optimization in parallel.
        If the function times out, do not save the evaluation.

        Args:
            num_function_calls (int): number of function calls
            nprocs (int): number of processes to use
        """
        if nprocs is None:
            nprocs = cpu_count()
        
        assert nprocs > 0, "nprocs must be positive"

        assert nprocs < num_function_calls, "Must have more evaluations than processes"

        start_opt = time.time()

        # start_print = len(self.saved_evaluations) + num_function_calls - nprocs

        # we need to save the candidates because the Queue will not be able to pickle them
        candidates = np.full(num_function_calls, None, dtype=object)

        candidates_t = np.zeros(num_function_calls)
        evals_t = np.zeros(num_function_calls)
        set_t = np.zeros(num_function_calls)

        # idea from https://github.com/tsoernes/gfsopt/blob/master/gfsopt/gfsopt.py#L321
        result_queue = Queue()

        def spawn_process(pid):
            start_t = time.time()
            # get candidate
            candidate = self.get_candidate()
            candidates_t[pid] = time.time() - start_t
            # print every 100 candidates
            # if pid%100 == 99: 
            #     print(f'Candidate {pid} time: {time.time()-start_t :.2f} s; Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')
            # store candidate to be set with result later
            candidates[pid] = candidate

            evals_t[pid] = time.time()
            # spawn process
            Process(target=_proc_timeout, args=(self.function, result_queue, pid, candidate.x,)).start()

            return

        def store_result():
            # get result
            pid, idxs, y, th = result_queue.get(block=True, timeout=None)
            # if it is an unsuccessful evaluation
            if y is None:
                # do not store result
                return
            evals_t[pid] = time.time() - evals_t[pid]
            # save evaluation
            self.saved_evaluations.append((candidates[pid].x, idxs, y, th))
            # update search
            start_t = time.time()
            candidates[pid].set(y)
            set_t[pid] = time.time() - start_t
            # print the last nprocs results executed
            # if start_print <= len(self.saved_evaluations):
            #     print(f'Candidate {pid} time: Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')
            # print every 500 candidates
            if pid%500 == 0: 
                print(f'\t# of evaluations: {len(self.saved_evaluations)}')
                print(f'\t\tCandidate {pid} time: {candidates_t[pid] :.2f} s; eval time: {evals_t[pid] :.2f} s; set time: {set_t[pid] :.2f} s; Total time: {time.time()-start_opt :.2f} s')
            
            return

        # initialize
        for i in range(nprocs):
            spawn_process(i)
        
        # run
        for i in range(nprocs, num_function_calls):
            # store result
            store_result()
            # add task
            spawn_process(i)
        
        # finish
        for _ in range(nprocs):
            # store result
            store_result()

        return



    def run_parallel4(self, num_function_calls: int = 1, nprocs=None):
        """
        Run optimization in parallel.
        In this case instead of cancelling the process, we just re-spawn it.

        Args:
            num_function_calls (int): number of function calls
            nprocs (int): number of processes to use
        """
        if nprocs is None:
            nprocs = cpu_count()
        
        assert nprocs > 0, "nprocs must be positive"

        assert nprocs < num_function_calls, "Must have more evaluations than processes"

        # start_print = len(self.saved_evaluations) + num_function_calls - nprocs

        # start_opt = time.time()

        # we need to save the candidates because the Queue will not be able to pickle them
        candidates = np.full(num_function_calls, None, dtype=object)

        # idea from https://github.com/tsoernes/gfsopt/blob/master/gfsopt/gfsopt.py#L321
        result_queue = Queue()

        def spawn_process(pid, timeout=3600):
            # start_t = time.time()
            # get next candidate
            candidate = self.get_candidate()
            # print every 100 candidates
            # if pid%100 == 99:
            #     print(f'Candidate {pid} time: {time.time()-start_t :.2f} s; Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')
            # store candidate to be set with result later
            candidates[pid] = candidate
            # spawn process
            Process(target=_proc_respawn, args=(self.function, result_queue, pid, candidate.x, timeout)).start()

            return

        def store_result():
            # get result
            pid, idxs, y, th, timeout = result_queue.get(block=True, timeout=None)
            # if it is an unsuccessful evaluation, we need to spawn a new process
            if y is None:
                half_timeout = timeout/2
                if half_timeout < 1:
                    # do not store result
                    return True
                else:
                    # re-spawn process with half the timeout
                    spawn_process(pid, timeout=timeout/2)
                    # wait for result
                    return False
            # save evaluation
            self.saved_evaluations.append((candidates[pid].x, idxs, y, th))
            # update search
            candidates[pid].set(y)
            # print the last nprocs results executed
            # if start_print <= len(self.saved_evaluations):
            #     print(f'Candidate {pid} time: Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')

            # good result
            return True

        # initialize
        for i in range(nprocs):
            spawn_process(i)
        
        # run
        for i in range(nprocs, num_function_calls):
            # we need to wait for finished processes
            while not store_result():
                pass
            # add task
            spawn_process(i)
        
        # finish
        for _ in range(nprocs):
            while not store_result():
                pass

        return



    def run_parallelX(self, num_function_calls: int = 1, nprocs=None):
        """
        run optimization in parallel

        Args:
            num_function_calls (int): number of function calls
            nprocs (int): number of processes to use
        """
        if nprocs is None:
            nprocs = cpu_count()
        
        assert nprocs > 0, "nprocs must be positive"

        assert nprocs < num_function_calls, "Must have more evaluations than processes"

        start_opt = time.time()

        # we need to save the candidates because the Queue will not be able to pickle them
        # candidates = np.full(num_function_calls, None, dtype=object)
        
        # pre-allocate num_iterations elements in the shared list
        self.shared_attribute.extend([None] * num_function_calls)

        # idea from https://github.com/tsoernes/gfsopt/blob/master/gfsopt/gfsopt.py#L321
        result_queue = Queue()

        # create a lock
        lock = Lock()

        def spawn_process(pid):
            start_t = time.time()
            candidate = self.get_candidate_par(lock)
            if pid%100 == 99: # print every 100 candidates
                print(f'Candidate {pid} time: {time.time()-start_t :.2f} s; Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')
            # evaluate function
            idxs,y,th = self.function(**candidate.x)
            # update search
            candidate.set(y)
            # save evaluation
            result_queue.put((pid, candidate.x, idxs, y, th))

            return
        
        def spawn_get(pid, shared_attribute):
            # print lower bounds
            print(self.lower_bounds)
            # print shared attribute length
            print(len(shared_attribute))
            candidate = self.get_candidate()
            shared_attribute[pid] = candidate
            result_queue.put(pid)
            return

        def store_result():
            # get result
            pid, x, idxs, y, th = result_queue.get(block=True, timeout=None)
            # print(f'Candidate {pid} time: Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')
            # save evaluation
            self.saved_evaluations.append((x, idxs, y, th))
            # print the last nprocs results executed
            # if num_function_calls-nprocs <= len(self.saved_evaluations):
            #     print(f'Candidate {pid} time: Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')

        def eval_store():
            # get result
            pid = result_queue.get(block=True, timeout=None)
            # evaluate function
            idxs,y,th = self.function(**self.shared_attribute[pid].x)
            # update search
            self.shared_attribute[pid].set(y)
            # save evaluation
            self.saved_evaluations.append((self.shared_attribute[pid].x, idxs, y, th))
            # print the last nprocs results executed
            # if num_function_calls-nprocs <= len(self.saved_evaluations):
            #     print(f'Candidate {pid} time: Total time: {time.time()-start_opt :.2f} s; # of evaluations: {len(self.saved_evaluations)}')

        for i in range(num_function_calls):
            p = Process(target=spawn_get, args=(i,self.shared_attribute))
            p.start()
            p.join()
            eval_store()

        # # initialize
        # for i in range(nprocs):
        #     # spawn process
        #     Process(target=spawn_process, args=(i,)).start()

        # # run
        # for i in range(nprocs, num_function_calls):
        #     # store result
        #     store_result()
        #     # spawn process
        #     Process(target=spawn_process, args=(i,)).start()

        # # finish
        # for _ in range(nprocs):
        #     # store result
        #     store_result()

        
        # for i in range(num_function_calls):
        #     p = Process(target=spawn_process, args=(i,))
        #     p.start()
        #     p.join()
        #     store_result()
        
        return


    def run(self, num_function_calls: int = 1):
        """
        run optimization

        Args:
            num_function_calls (int): number of function calls
        """
        for i in range(num_function_calls):
            candidate = self.get_candidate()
            # evaluate
            # idxs: list of pattern indices
            # y: DR score
            # th: threshold
            idxs,y,th = self.function(**candidate.x)
            # save evaluation
            self.saved_evaluations.append((candidate.x, idxs, y, th))
            # update search
            candidate.set(y)

    @property
    def running_optimum(self):
        """
        maximum by evaluation step

        Returns:
            list: value of optimum for each evaluation
        """
        optima = []
        for e in self.evaluations:
            # e = (dict, idxs, val, th)
            if len(optima) == 0:
                optima.append(e[2])
            else:
                if self.maximize:
                    if optima[-1] > e[2]:
                        optima.append(optima[-1])
                    else:
                        optima.append(e[2])
                else:
                    if optima[-1] < e[2]:
                        optima.append(optima[-1])
                    else:
                        optima.append(e[1])
        return optima

# Handle multiprocessing
def _dlib_proc(obj_func, result_queue, pid, x):
    # evaluate
    idxs,y,th = obj_func(**x)
    # save evaluation
    result_queue.put((pid, idxs, y, th))

    return

def _proc_timeout(func, q, pid, x, timeout=3600):

    def evaluate():
        
        # handle SIGTERM signal
        def sigterm_handler(signum, frame):
            # save unfinished evaluation
            q.put((pid, None, None, None))
            # exit process
            sys.exit(0)

        # register SIGTERM signal handler
        signal.signal(signal.SIGTERM, sigterm_handler)
        # evaluate
        idxs,y,th = func(**x)
        # save evaluation
        q.put((pid, idxs, y, th))

        return
    
    # start process
    p = Process(target=evaluate)
    p.start()
    # join process with timeout
    p.join(timeout=timeout)
    # if process is still active
    if p.is_alive():
        # terminate process
        p.terminate()
        # join process
        p.join()
        print(f'Process {p.pid} took too long. Evaluation {pid} was skipped.')

    return

def _proc_respawn(func, q, pid, x, timeout=3600):

    def evaluate():
        
        # handle SIGTERM signal
        def sigterm_handler(signum, frame):
            # save unfinished evaluation
            q.put((pid, None, None, None, timeout))
            # exit process
            sys.exit(0)

        # register SIGTERM signal handler
        signal.signal(signal.SIGTERM, sigterm_handler)
        # evaluate
        idxs,y,th = func(**x)
        # save evaluation
        q.put((pid, idxs, y, th, timeout))

        return
    
    # start process
    p = Process(target=evaluate)
    p.start()
    # join process with timeout
    p.join(timeout=timeout)
    # if process is still active
    if p.is_alive():
        # terminate process
        p.terminate()
        # join process
        p.join()
        print(f'Process {p.pid} took too long. Evaluation {pid} was skipped. New timeout: {timeout//2} s.')

    return
