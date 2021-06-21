# ilp-converter
Converts LPs or MIPs to standard form. 

## usage

See example_usage.py

## Dependencies
```
numpy, scip, mip
```
## Features

* Currently only [Python-MIP](https://python-mip.com/) file reader is used as the reader of `.mps` files. See the documentation of Python-MIP for a list of all supported file formats. All formats supported by Python-MIP are also supported by this code.

* Alternatively, you can build a `python-mip` `Model` and give that to the standard form converter. See `example_usage.py` for an example.
