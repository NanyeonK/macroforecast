# Security

## Reporting a vulnerability

Report privately through GitHub's
[security advisory form](https://github.com/NanyeonK/macroforecast/security/advisories/new),
not as a public issue. A first response should arrive within a week.

Please include what you were running, what happened, and — if you have one — a
minimal reproduction. A report without a reproduction is still worth sending.

## What this package assumes about the files it reads

`macroforecast` is a research library that runs on your own machine against data
you supply. Two of the things it reads carry a trust assumption worth stating
explicitly, because neither is obvious from the call site.

### Pickles execute code

`load_fit()` unpickles a saved model, and the preprocessing cache and result
store unpickle their entries. **A pickle file is a program, not a data format:
loading one can run arbitrary code before it returns anything.** No amount of
inspecting the file first makes this safe.

So:

- Load only artifacts you produced, or that came from a source you trust as much
  as you trust your own machine.
- Treat a cache or store directory as trusted: never point one at a location
  another party can write to, and never share one across a trust boundary.
- To hand fitted state to someone else, ship the code and specification that
  reproduce it, not the pickle. That is also what makes the result checkable.

The sidecar JSON written next to a saved model is plain data and carries no such
risk; read it when you only need the metadata.

### User-supplied callables run as your code

`custom_step`, `custom_preprocess_step`, custom models, and custom loss functions
are called directly. They run with your privileges. This is intended — it is how
the package stays open to methods it does not ship — but it means a specification
you received from someone else is code you are about to execute, and deserves the
same reading you would give a script.

## Scope

In scope: anything letting a *data* file — a CSV, a FRED download, a panel —
cause code execution or escape its expected effect on results.

Expected execution of a pickle the caller explicitly supplied, or of a callable the
caller explicitly passed, is not by itself a vulnerability — the caller chose the
input, and the docs say what it costs.

In scope, and please report:

- **unexpected unpickling** — deserialization the caller did not ask for
- **loading from a path the caller did not select** — an implicit search path, a
  default directory, an environment variable
- **path or symlink substitution** in cache, model-store or artifact directories
- **trust-boundary bypass** — untrusted input reaching a trusted deserializer
- **code execution triggered by a format documented as data-only** (CSV, Parquet,
  JSON, a manifest)
which are documented properties rather than defects. If you can turn one of them
into a surprise for someone following the documentation, that *is* in scope, and
worth reporting.
