# Contributing

Install with `python -m pip install -e '.[train,fk,dev]'`, then run
`python -m pytest -q`. Open an issue with the command, Python/PyTorch versions,
robot joint order and coordinate convention needed to reproduce a problem.
Do not attach credentials or private demonstration data.

Model changes should preserve checkpoint compatibility or explicitly version
the bundle format. Add a regression test for changes to history handling,
normalization, quaternion conventions, FK losses or evaluation aggregation.
Report the exact checkpoint and data split when sharing experiment results.
