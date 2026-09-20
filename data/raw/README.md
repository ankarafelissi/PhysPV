# Raw Data

Store source HDF5 datasets in this directory. Configure the input file with
`data.path` and its observation interval with `data.resolution`.

- `SOLETE_short.h5` is the tracked 24-row hourly example used for functional checks.
- Full-resolution datasets are local inputs and are excluded from version control.
- Source files are read without modification. Generated features belong in
  `data/processed/`, and experiment results belong in `outputs/`.

See [the data contract](../../docs/data.md) for schema, feature requirements,
and temporal handling. Third-party dataset and code attribution is also recorded
there.
