# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python quantitative research project. The runnable code lives in the
`crypto_lab` package; the top-level `NN-*.py` files are reference JoinQuant strategy
sources that only run on the joinquant.com platform (do not execute them locally).

- Dependencies (numpy, matplotlib) are declared in `pyproject.toml` and installed by the
  update script (`pip install -e .`). matplotlib is used with a headless backend, so no
  display is required.
- Run the app with the module form: `python3 -m crypto_lab.cli <subcommand>`. The
  `crypto-lab` console script installs to `~/.local/bin`, which is not on `PATH`, so the
  module form is the reliable entry point.
- Subcommands and their standard usage are documented in `README.md`. The
  `research`, `optimize`, `crypto-alpha`, and `cycle-report` commands run fully offline
  against the committed CSV snapshots under `data/` — no network is needed.
- Only the `download` subcommand (and `ema-research` / `cta-research` with `--refresh`)
  hit the OKX public API. Outbound network may be blocked in this environment; prefer the
  cached-data commands above for verification.
- Tests: `python3 -m unittest discover -s tests -v` (37 tests, runs in well under a second).
- No linter/formatter is configured. For a quick syntax sanity check use
  `python3 -m compileall crypto_lab tests`.
- Report commands overwrite generated artifacts under `reports/` (markdown is
  deterministic; PNG charts differ only in binary metadata on re-render). Use
  `git checkout -- reports/` to discard incidental regenerated-chart diffs.
- The tracked working-tree diff on the `NN-*.py` files is only CRLF→LF normalization from
  `.gitattributes`; leave it alone (do not commit line-ending churn).
