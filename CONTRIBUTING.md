# Contributing to gfem

Thanks for your interest. gfem is young: its API still changes between releases
and its design moves quickly. Contributions are welcome, within the limits below.

## What helps most right now

- **Bug reports.** A small script that runs on its own, the output you expected
  and the output you got, and your platform and gfem version (or commit).
- **Validation cases.** gfem compared with another solver, an analytic result or
  a measurement. Keep the case in your own repository and depend on gfem from
  there. When gfem disagrees, or a feature is missing or awkward to use, open an
  issue with the smallest reproducer you can make. Strong cases may be brought
  into gfem's test suite, with credit to you.
- **Documentation fixes**: typos, unclear steps, broken links.

## Before writing code

Open an issue first and wait for a reply before starting a pull request,
especially for anything larger than a small fix. The core is moving quickly, so a
change may conflict with work in progress, and an idea may be implemented in a
different way, or declined. Talking it over first saves your time.

## Pull requests

1. Fork the repository, branch from `main`, and link the pull request to its
   issue. Draft pull requests are fine.
2. Set up and run the checks that CI runs:

   ```bash
   uv sync
   uv run ruff check --no-fix . && uv run ruff format --check .
   uv run ty check src
   uv run pytest -m "unit or verification"
   ```

   [Development](https://bvial.info/gfem/project/development.html) covers the
   test tiers, the optional PETSc, MPI and GPU stacks, and building the docs.
3. Write formulas as LaTeX math, never as plain text: `$…$` in Markdown and
   notebooks, `` :math:`…` `` in docstrings, and make a docstring that holds a
   backslash a raw string (`r"""…"""`).
4. Add a test for each change in behaviour, and keep each pull request to one
   change.

## Licence

gfem is under the [MIT licence](LICENSE), and contributions are accepted under
the same licence.

## Coding agents

You may use AI coding tools. You are responsible for every line you submit:
read it, run it, and be ready to explain it. The maintainer's design notes and
agent configuration are kept outside this repository; nothing here needs them.
