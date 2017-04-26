Smuggler: read STATA, SPSS, SAS data sets from Python
===

Smuggler enables you to load the data sets of stats packages as pandas DataFrame using [ReadStat](https://github.com/WizardMac/ReadStat) C library written by [Evan Miller](http://www.evanmiller.org).

Supported formats are below.

- STATA dta file
- SPSS sav file
- SPSS por file
- SAS sas7bdat file

Some of them are already supported in pandas, so I want to compare the performance.

SAS catalog format (.sas7bcat) is unavailable though ReadStat library can read them.


Installation
---

```bash
$ python setup.py build install
```

Requirement
---

- pandas
- numpy

Usage
---

- STATA dta file: read_dta("path/to/file")
- SPSS sav file: read_sav("path/to/file")
- SPSS por file: read_por("path/to/file")
- SAS sas7bdat file: read_sas7bdat("path/to/file")


TODO
---

- Make test suites (currently I just checked with a few files).
- Support reading RDS and Rdata
- Support writing dta and sav
- Create a wheel for Windows
