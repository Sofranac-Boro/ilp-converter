import argparse, sys, traceback

import mip
from scipy.sparse import csr_matrix

from StandardFormConverter import StandardFormConverter
from readerInterface import FileReaderInterface, get_reader
from problemInterface import ProblemInterface, get_problem


def csr_to_csc(m, n, vals, col_indices, row_ptrs):
    A = csr_matrix((vals, col_indices, row_ptrs), shape=(m, n)).tocsc()
    return A.data, A.indices, A.indptr


def from_mip_model() -> None:

    n_cons = 3
    n_vars = 4

    m = mip.Model()
    x1 = m.add_var("x1", 2,100, 2)
    x2 = m.add_var("x2", 2,mip.INF, 5)
    x3 = m.add_var("x3", 2,mip.INF, -1)
    x4 = m.add_var("x4", -2,mip.INF, -1)

    m += x1 + x2 + x3 + x4 <= 10
    m +=         5*x3 + x4 >= 1
    m += 5*x1 +  1*x3 + 1*x4 >= 5

    reader: FileReaderInterface = get_reader("", model=m)

    # to read from file, just use:
    # reader: FileReaderInterface = get_reader(path_to_file)

    coeffs, row_ptrs, col_indices = reader.get_cons_matrix()
    lbs, ubs = reader.get_var_bounds()
    senses = reader.get_senses()
    rhss = reader.get_rhss()
    costs = reader.get_costs()

    prb: ProblemInterface = get_problem(n_cons, n_vars, coeffs, col_indices, row_ptrs, lbs, ubs, senses, rhss, costs)
    converter: StandardFormConverter = StandardFormConverter(prb)

    print("A before:\n", converter.prb.A)
    converter.to_standard_form()
    print("A after:\n", converter.prb.A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve LP file via criss-cross')
    parser.add_argument("-f", "--file", type=str, required=False)
    args = parser.parse_args()

    try:
        from_mip_model()
    except Exception as e:
        print("\nexecution of ", args.file, " failed. Exception: ")
        print(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, file=sys.stdout)