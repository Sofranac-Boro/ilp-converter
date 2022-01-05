# ilp-converter
Reads a LP or a MIP from a file and converts it to standard form.

* Currently only [Python-MIP](https://python-mip.com/) file reader is used as the reader of `.mps` files. See the documentation of Python-MIP for a list of all supported file formats. All formats supported by Python-MIP are also supported by this code.

* Alternatively, you can build a Python-MIP `Model` and give that to the standard form converter. See `example_usage.py` for an example.

## usage

```
python3 convert.py -f "23588.mps.gz"
```

## Dependencies
```
numpy, scipy, mip
```

