import abc

import sinter


class CollectionWorkHandler(metaclass=abc.ABCMeta):
    """Defines the unit of task-specific work executed by collection workers."""

    @abc.abstractmethod
    def do_some_work(self, task: sinter.Task, max_shots: int) -> sinter.AnonTaskStats:
        """Processes at most ``max_shots`` from ``task`` and returns the delta."""

        pass
