# Local controller qualification: incomplete

The finite-world diagnostic passed its assertions; exact trajectories and results are under worlds/. The aggregate regression suite was invalid at fixture setup: the runner supplied a nested pytest --basetemp without creating its parent. 29 tests passed and 81 fixture setups errored before exercising their cases. No controller conclusion follows from those setup errors. Fix only output-directory preparation and run the regression suite under a fresh card. Original outputs are retained.
