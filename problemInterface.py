from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
from typing import List, Tuple
from enum import Enum
from utils import INF, EPSGE, EPSLE, EPSEQ, flatten_list
from abc import ABC, abstractmethod


class Sense(Enum):
    LE = "<"
    GE = ">"
    EQ = "="
    EMPTY = ""


class ProblemInterface(ABC):

    @abstractmethod
    def add_cons(self, sense: Sense, rhs: float) -> int:
        pass

    @abstractmethod
    def set_coeff(self, cons_idx: int, var_idx: int, val: float):
        pass

    @abstractmethod
    def add_column(self) -> int:
        pass

    @abstractmethod
    def is_ub_inf(self, var_idx: int) -> bool:
        pass

    @abstractmethod
    def is_lb_inf(self, var_idx: int) -> bool:
        pass

    @abstractmethod
    def is_lb_zero(self, var_idx: int) -> float:
        pass

    @abstractmethod
    def set_ub(self, var: int, val: float):
        pass

    @abstractmethod
    def set_lb(self, var: int, val: float):
        pass

    @abstractmethod
    def get_ub(self, var: int) -> float:
        pass

    @abstractmethod
    def get_lb(self, var: int) -> float:
        pass

    @abstractmethod
    def get_coeff(self, cons: int, var: int) -> float:
        pass

    @abstractmethod
    def get_cost(self, var: int) -> float:
        pass

    @abstractmethod
    def set_cost(self, var:int, val: float):
        pass

    @abstractmethod
    def get_sense(self, cons: int) -> Sense:
        pass

    @abstractmethod
    def set_sense(self, cons: int, val: Sense):
        pass

    @abstractmethod
    def get_rhs(self, cons: int) -> float:
        pass

    @abstractmethod
    def set_rhs(self, cons: int, val: float):
        pass

    @abstractmethod
    def check_problem_validity(self, slack_basis: List[int]) -> None:
        pass

    @abstractmethod
    def to_csc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def copy_cons(self, cons: int) -> int:
        pass


class ProblemDense(ProblemInterface):
    def __init__(self, ncons: int, nvars: int, coeffs: List[float], col_indices: List[float], row_ptrs: List[float], lbs: List[float], ubs: List[float], senses: List[Sense], rhss: List[float], costs: List[float]):
        self.ncons = ncons
        self.nvars = nvars

        self.A = csr_matrix((coeffs, col_indices, row_ptrs), shape=(ncons, nvars)).todense()
        self.lbs = lbs
        self.ubs = ubs
        self.senses = senses
        self.b = rhss
        self.costs = costs

    def add_cons(self, sense: Sense, rhs: float) -> int:
        self.A=np.vstack([self.A, np.zeros(self.nvars)])
        self.ncons += 1

        self.senses.append(sense)
        self.b.append(rhs)

        return self.ncons -1 # index of the added cons

    def copy_cons(self, cons: int) -> int:
        new_cons_idx = self.add_cons(self.get_sense(cons), self.get_rhs(cons))

        for var in range(self.nvars):
            self.set_coeff(new_cons_idx, var, self.get_coeff(cons, var))

        return new_cons_idx # return index of the new cons

    def set_coeff(self, cons_idx: int, var_idx: int, val: float):
        self.A[cons_idx, var_idx] = val

    def add_column(self) -> int:
        coeffs = np.zeros((self.ncons, 1))
        self.A = np.hstack([self.A, coeffs])

        self.costs.append(0)

        self.ubs.append(INF)
        self.lbs.append(0.0)

        self.nvars += 1
        return self.nvars-1 # index of the added column

    def is_ub_inf(self, var_idx: int) -> bool:
        return EPSGE(self.ubs[var_idx], INF)

    def is_lb_inf(self, var_idx: int) -> bool:
        return EPSLE(self.lbs[var_idx], -INF)

    def is_lb_zero(self, var_idx: int) -> float:
        return EPSEQ(self.lbs[var_idx], 0.0)

    def set_ub(self, var: int, val: float):
        self.ubs[var] = val

    def set_lb(self, var: int, val: float):
        self.lbs[var] = val

    def get_ub(self, var: int) -> float:
        return self.ubs[var]

    def get_lb(self, var: int) -> float:
        return self.lbs[var]

    def get_coeff(self, cons: int, var: int) -> float:
        return self.A[cons, var]

    def get_cost(self, var: int) -> float:
        return self.costs[var]

    def set_cost(self, var:int, val: float):
        self.costs[var] = val

    def get_sense(self, cons: int) -> Sense:
        return self.senses[cons]

    def set_sense(self, cons: int, val: Sense):
        self.senses[cons] = val

    def get_rhs(self, cons: int) -> float:
        return self.b[cons]

    def set_rhs(self, cons: int, val: float):
        self.b[cons] = val

    def check_problem_validity(self, slack_basis: List[int]) -> None:
        assert len(self.ubs) == self.nvars
        assert len(self.lbs) == self.nvars
        assert len(self.costs) == self.nvars
        assert self.A.shape[0] == self.ncons
        assert self.A.shape[1] == self.nvars
        assert len(self.b) == self.ncons
        assert len(self.senses) == self.ncons

        for sense in self.senses:
            assert sense == Sense.EQ

        for var in range(self.nvars):
            assert self.is_ub_inf(var)
            assert self.is_lb_zero(var)

        self.check_for_zero_cols()
        self.check_if_slack_basis_is_identity(slack_basis)

    def check_for_zero_cols(self) -> None:
        for col in range(self.nvars):
            assert any(flatten_list(self.A[:,col]))

    def check_if_slack_basis_is_identity(self, slack_basis: List[int]) -> None:
        basis = self.A[:,slack_basis]

        for i in range(self.ncons):
            for j in range(self.ncons):
                if i == j:
                    assert np.isclose(1.0, abs(basis[i, j])) # can be + or - 1
                else:
                    assert np.isclose(0.0, basis[i, j])

    def to_csc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        csc = csc_matrix(self.A)
        return csc.data, csc.indices, csc.indptr


def get_problem(
                ncons: int,
                nvars: int,
                coeffs: List[float],
                col_indices: List[float],
                row_ptrs: List[float],
                lbs: List[float],
                ubs: List[float],
                senses: List[Sense],
                rhss: List[float],
                costs: List[float]
                ) -> ProblemInterface:
    return ProblemDense(ncons, nvars, coeffs, col_indices, row_ptrs, lbs, ubs, senses, rhss, costs)