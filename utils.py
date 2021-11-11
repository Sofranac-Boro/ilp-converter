EPS=1e-9
INF=1e20


def EPSLE(x, y):
    return x-y <= EPS
def EPSLT(x, y):
    return x-y < -EPS
def EPSGE(x, y):
    return x-y >= -EPS
def EPSGT(x, y):
    return x-y > EPS
def EPSEQ(x, y):
    return abs(x-y) <= EPS

def flatten_list(l):
    return [item for sublist in l for item in sublist]
