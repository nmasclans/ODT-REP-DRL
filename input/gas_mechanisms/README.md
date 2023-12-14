## Cantera Compatibility issues

New versions of Cantera no longer accept `.xml` or `.cti` input files, but only `.yaml` files.
For this purpose, `not_used.xml` has been converted to `not_used.yaml` using the integreted Cantera command:
```
ctml2yaml not_used.xml
```

__Note:__ prior to the conversion with `ctml2yaml`, the original line in `not_used.xml` regarding the `acentric_factor` is commented for avoinding `TypeError`. 

