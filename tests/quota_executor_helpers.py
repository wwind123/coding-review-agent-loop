"""Deterministic executor scheduling without real thread timing."""
from concurrent.futures import Future


class ScriptedExecutor:
    def __init__(self, *, max_workers, thread_name_prefix):
        self.futures = []
        self.consumed = []
        self.shutdown_calls = []

    def submit(self, fn, *args):
        owner = self
        class RecordedFuture(Future):
            def result(self, *a, **kw):
                assert self.done(), 'unbounded result observation before completion'
                owner.consumed.append(args[-1])
                return super().result(*a, **kw)
        future = RecordedFuture()
        future.execute = lambda: fn(*args)
        self.futures.append(future)
        return future

    def complete(self, index, *, error=None):
        future = self.futures[index]
        if error is not None:
            future.set_exception(error)
        else:
            try:
                future.set_result(future.execute())
            except BaseException as exc:
                future.set_exception(exc)

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)
