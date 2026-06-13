# Package Data

This directory contains non-code data files that should ship with QMHub when
they are needed at runtime or by tests.

Please note that it is not recommended to place large files in your git directory. If your project requires files larger
than a few megabytes in size it is recommended to host these files elsewhere. This is especially true for binary files
as the `git` structure is unable to correctly take updates to these files and will store a complete copy of every version
in your `git` history which can quickly add up. As a note most `git` hosting services like GitHub have a 1 GB per repository
cap.

## Including package data

Package data is configured through `pyproject.toml` and `MANIFEST.in`. Add wheel
package-data entries under `[tool.setuptools.package-data]` in `pyproject.toml`
when runtime imports need the file, and keep source-distribution include/exclude
rules in `MANIFEST.in` in sync.

## Manifest

* `look_and_say.dat`: first entries of the "Look and Say" integer series, sequence [A005150](https://oeis.org/A005150)
