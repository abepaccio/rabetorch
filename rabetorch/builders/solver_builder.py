import torch

from rabetorch.solvers import SGD

_solver = {
    "SGD": SGD
}

class SolverBuilder():
    def __init__(self, _solver_cfg) -> None:
        self.solver_cfg = _solver_cfg
        self._solver = _solver[_solver_cfg.OPTIMIZER]
        self.max_epoch = _solver_cfg.MAX_EPOCH

    def build_solver(self, model):
        solver = self._solver(self.solver_cfg)
        return solver.get_solver(model), self.max_epoch
