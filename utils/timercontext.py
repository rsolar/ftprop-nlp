from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def timer_context(name, timers_dict):
    start_time = default_timer()

    yield

    if name not in timers_dict:
        timers_dict[name] = 0.0
    timers_dict[name] += default_timer() - start_time
