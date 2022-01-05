from abc import ABC, abstractmethod
from typing import Tuple, List, Generator, Dict
import shutil
import tempfile
import os

from mip import Model, LinExpr, Var
from problemInterface import Sense


class FileReaderInterface(ABC):

    @abstractmethod
    def get_n_vars(self) -> int:
        pass

    @abstractmethod
    def get_n_cons(self) -> int:
        pass

    @abstractmethod
    def get_nnz(self) -> int:
        pass

    @abstractmethod
    def get_var_bounds(self) -> Tuple[List[float], List[float]]:
        pass

    @abstractmethod
    def get_lrhss(self) -> Tuple[List[float], List[float]]:
        pass

    @abstractmethod
    def get_cons_matrix(self) -> Tuple[List[float], List[int], List[int]]:
        pass

    @abstractmethod
    def get_SCIP_vartypes(self) -> List[int]:
        pass

    @abstractmethod
    def write_model_with_new_bounds(self, lbs: List[float], ubs: [List[float]]) -> str:
        pass

    @abstractmethod
    def get_costs(self) -> List[float]:
        pass

    @abstractmethod
    def is_minimization_problem(self) -> bool:
        pass

    @abstractmethod
    def get_senses(self) -> List[Sense]:
        pass

    @abstractmethod
    def get_rhss(self) -> List[float]:
        pass

    @abstractmethod
    def solve(self) -> None:
        pass


class PythonMIPReader(FileReaderInterface):
    def __init__(self, input_file: str, model: Model = None) -> None:

        if model is not None:
            self.m = model
        else:
            self.instance_name = input_file.split("/")[-1].split(".")[0]
            self.m = Model()

            print("Reding lp file", self.instance_name)
            self.m.read(input_file)
            print("Reading of ", self.instance_name, " model done!")

        self.tmp_reader_dir = tempfile.mkdtemp()

    def __del__(self):
        shutil.rmtree(self.tmp_reader_dir)

    def is_minimization_problem(self) -> bool:
        return self.m.sense == "MIN"

    def write_model_with_new_bounds(self, lbs: List[float], ubs: [List[float]]) -> str:
        assert(len(lbs) == len(ubs) == len(self.m.vars))
        for i in range(len(lbs)):
            self.m.vars[i].lb = lbs[i]
            self.m.vars[i].ub = ubs[i]
        out_path = os.path.join(self.tmp_reader_dir, str(self.instance_name) + ".mps")
        self.m.write(out_path)
        return out_path

    def get_n_vars(self) -> int:
        return self.m.num_cols

    def get_n_cons(self) -> int:
        return self.m.num_rows

    def get_nnz(self) -> int:
        return self.m.num_nz

    def get_var_bounds(self) -> Tuple[List[float], List[float]]:
        ubs = map(lambda var: var.ub, self.m.vars)
        lbs = map(lambda var: var.lb, self.m.vars)
        return list(lbs), list(ubs)

    def get_costs(self) -> List[float]:
        return list(map(lambda var: var.obj, self.m.vars))

    def get_lrhss(self) -> Tuple[List[float], List[float]]:
        lhss = map(
            lambda cons: float('-Inf') if cons.expr.sense == '<' else cons.rhs,
            self.m.constrs
        )
        rhss = map(
            lambda cons: float('Inf') if cons.expr.sense == '>' else cons.rhs,
            self.m.constrs
        )
        return list(lhss), list(rhss)

    def get_senses(self) -> List[Sense]:
        conversion_dict = {"<": Sense.LE, ">": Sense.GE, "=": Sense.EQ, "" : Sense.EMPTY}
        senses = map(lambda cons: cons.expr.sense, self.m.constrs)
        return list(map(
            conversion_dict.get,
            senses
        ))

    def get_rhss(self):
        return list(map(lambda cons: cons.rhs, self.m.constrs))

    def get_cons_matrix(self) -> Tuple[List[float], List[int], List[int]]:

        def get_expr_coos(expr: LinExpr, var_indices: Dict[Var, int]) -> Generator:
            for var, coeff in expr.expr.items():
                yield coeff, var_indices[var]

        row_indices = []
        row_ptrs = []
        col_indices = []
        coeffs = []

        var_indices = {v: i for i, v in enumerate(self.m.vars)}

        row_ctr = 0
        row_ptrs.append(row_ctr)

        for row_idx, constr in enumerate(self.m.constrs):

            for coeff, col_idx in get_expr_coos(constr.expr, var_indices):
                row_ctr += 1
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                coeffs.append(coeff)

            row_ptrs.append(row_ctr)

        return coeffs, row_ptrs, col_indices

    def get_SCIP_vartypes(self) -> List[int]:
        conversion_dict = {'B': 0, 'I': 1, 'C': 3}
        python_mip_vartypes = map(lambda var: var.var_type, self.m.vars)
        return list(map(
            conversion_dict.get,
            python_mip_vartypes
        ))


    def solve(self) -> None:
        assert self.is_minimization_problem()
        self.m.optimize()


# Instantiation
def get_reader(input_file: str, model: Model = None) -> FileReaderInterface:
    return PythonMIPReader(input_file, model)

