import unittest
from lib.ilp_converter.StandardFormConverter import StandardFormConverter
from lib.ilp_converter.readerInterface import FileReaderInterface, get_reader
from lib.ilp_converter.problemInterface import ProblemInterface, get_problem
from mip import Model, INF
from utils import EPSEQ
import numpy as np

from typing import List, Union


def list_equal(list1: Union[List, np.ndarray], list2: Union[List, np.ndarray]) -> bool:
    for idx in range(len(list1)):
        if not EPSEQ(list1[idx], list2[idx]):
            return False
    return True


class MyTestCase(unittest.TestCase):
    def test_LiExample(self):
        n_cons = 4
        n_vars = 4
        nnz = 10
        m = Model()

        x1 = m.add_var("x1", 0, INF, 3)
        x2 = m.add_var("x2", 0, INF, 5)
        x3 = m.add_var("x3", 0, INF, -1)
        x4 = m.add_var("x4", 0, INF, -1)

        m += x1 +  x2 + x3 + x4 <= 40
        m +=5*x1 + x2           <= 12
        m +=            x3 + x4 >= 5
        m +=            x3 + 5*x4 <= 50

        reader: FileReaderInterface = get_reader("", model=m)

        coeffs, row_ptrs, col_indices = reader.get_cons_matrix()
        lbs, ubs = reader.get_var_bounds()
        senses = reader.get_senses()
        rhss = reader.get_rhss()
        costs = reader.get_costs()

        prb: ProblemInterface = get_problem(n_cons, n_vars, coeffs, col_indices, row_ptrs, lbs, ubs, senses, rhss, costs)

        converter = StandardFormConverter(prb)
        converter.to_standard_form()

        A_sol = [[1.0,    1.0,    1.0,    1.0,    -1.0,    0.0,    0.0,    0.0],

                  [5.0,    1.0,    0.0,    0.0,    0.0,    -1.0,    0.0,    0.0],

                  [0.0,    0.0,    1.0,    1.0,    0.0,    0.0,    1.0,    0.0],

                  [0.0,    0.0,    1.0,    5.0,    0.0,    0.0,    0.0,    -1.0 ]]

        for row in range(converter.prb.ncons):
            for col in range(converter.prb.nvars):
                self.assertTrue(EPSEQ(A_sol[row][col], converter.prb.A[row, col]))
            print("\n")

        vals_sol =         [1, 5, 1, 1, 1, 1, 1, 1, 1, 5, -1, -1, 1, -1]
        row_indices_sol =  [0, 1, 0, 1, 0, 2, 3, 0, 2, 3, 0, 1, 2, 3]
        col_ptrs_sol    =  [0, 2, 4, 7, 10, 11, 12, 13, 14]

        vals, row_indices, col_ptrs = prb.to_csc()

        self.assertTrue(list_equal(vals_sol, vals))
        self.assertTrue(list_equal(row_indices_sol, row_indices))
        self.assertTrue(list_equal(col_ptrs_sol, col_ptrs))


if __name__ == '__main__':
    unittest.main()
