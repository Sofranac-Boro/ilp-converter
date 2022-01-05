import argparse, sys, traceback

import mip
from scipy.sparse import csr_matrix

from StandardFormConverter import StandardFormConverter
from readerInterface import FileReaderInterface, get_reader
from problemInterface import ProblemInterface, get_problem


def csr_to_csc(m, n, vals, col_indices, row_ptrs):
    A = csr_matrix((vals, col_indices, row_ptrs), shape=(m, n)).tocsc()
    return A.data, A.indices, A.indptr


def from_mip_model(path_to_file) -> None:

    reader: FileReaderInterface = get_reader(path_to_file)

    # Optinal. Solve the problem
    reader.solve()

    n_cons = reader.get_n_cons()
    n_vars = reader.get_n_vars()
    coeffs, row_ptrs, col_indices = reader.get_cons_matrix()
    lbs, ubs = reader.get_var_bounds()
    senses = reader.get_senses()
    rhss = reader.get_rhss()
    costs = reader.get_costs()

    prb: ProblemInterface = get_problem(n_cons, n_vars, coeffs, col_indices, row_ptrs, lbs, ubs, senses, rhss, costs)
    converter: StandardFormConverter = StandardFormConverter(prb)

    print("A before convering to LP standard form:\n", converter.prb.A)
    converter.to_standard_form()
    print("A after converting to LP standard form:\n", converter.prb.A)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve LP file via criss-cross')
    parser.add_argument("-f", "--file", type=str, required=False)
    args = parser.parse_args()

    try:
        from_mip_model(args.file)
    except Exception as e:
        print("\nexecution of ", args.file, " failed. Exception: ")
        print(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, file=sys.stdout)