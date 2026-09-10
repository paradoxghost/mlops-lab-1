# MLOps Lab 1 Answers

## Question 1

With uv 0.12.13, `uv init` detected the existing Git repository and created a packaged Python project containing:

- `.python-version`: selects Python 3.13 for uv when it resolves and runs the project environment.
- `pyproject.toml`: stores project metadata, the supported Python version, dependencies, the console-script entry point, and the uv build backend configuration.
- `README.md`: the project readme referenced by `pyproject.toml`; uv created it as an empty file.
- `src/mlops_lab_1/__init__.py`: defines the initial `main()` function used by the `mlops-lab-1` console-script entry point.

This version of `uv init` did not create `main.py` or a top-level `.gitignore` in this repository. `uv.lock` was created later when DVC was added as a reproducible project development dependency.

## Question 2

With DVC 3.67.1, `dvc init` created:

- `.dvc/config`: the repository-level DVC configuration. It was initially empty and is meant to be tracked by Git. Safe shared settings, such as the name of the default remote, belong here; credentials do not.
- `.dvc/.gitignore`: tells Git to ignore `.dvc/config.local`, `.dvc/tmp`, and `.dvc/cache`. The ignore file itself is tracked by Git.
- `.dvcignore`: contains DVC-specific ignore patterns, similar in purpose to `.gitignore` but used when DVC scans the workspace. Its generated content is only explanatory comments, and the file is tracked by Git.
- `.dvc/tmp/btime`: an internal runtime file under DVC's ignored temporary directory. It must not be tracked by Git.

The `.dvc` directory itself also holds ignored local state. `.dvc/cache` stores local cached data objects when present, `.dvc/tmp` stores temporary/internal state, and `.dvc/config.local` can hold repository-specific private settings. Those paths must not be committed. Git should track `.dvc/config`, `.dvc/.gitignore`, and `.dvcignore`, along with later `.dvc` pointer files such as `data.dvc`.

## Question 3

The DagsHub remote URL, basic-auth mode, username, and password were configured with `--global`. DVC reports the Windows global configuration location as `C:\Users\vitor\AppData\Local\iterative\dvc\config`. Because this installation uses Microsoft Store Python, Windows physically redirects that file to the Python package's `LocalCache\Local\iterative\dvc\config` location. It is outside this Git repository in either case. The expected global keys were verified as present without displaying their values.

DVC configuration scopes include:

- Repository/project scope (the default): `.dvc/config`, shared through Git. In this project it contains only the safe setting that selects `origin` as the default remote.
- Local scope (`--local`): `.dvc/config.local`, specific to this checkout and ignored by the generated `.dvc/.gitignore`.
- Global scope (`--global`): the current user's DVC configuration, shared by that user's repositories on this machine.
- System scope (`--system`): machine-wide configuration; DVC reports its Windows path as `C:\ProgramData\iterative\dvc\config`.

Passwords and tokens must never be pushed to GitHub because anyone with repository access could retrieve and abuse them, including from Git history after a later deletion. Therefore Git tracks the non-secret `.dvc/config`, while credential-bearing global or local configuration stays outside Git.

## Question 4

The first `dvc add data` created the repository's top-level `.gitignore` with exactly one rule: `/data`. The leading slash anchors the rule at the repository root. As a result, Git ignores the real files under the `data` directory, while Git tracks the small `data.dvc` pointer. DVC stores and versions the actual dataset content in its cache and configured remote instead of placing 16,643 images in Git.

## Question 5

The generated `data.dvc` contains one output entry with these actual fields:

- `md5: a3a457d03c51ff8b037a833440f6ad13.dir`: the content-addressed DVC hash for this directory version; the `.dir` suffix identifies a directory object.
- `size: 1188442712`: the total tracked size in bytes.
- `nfiles: 16643`: the number of files in this version.
- `hash: md5`: the hash algorithm used for the DVC object.
- `path: data`: the workspace path represented by the pointer.

Git versions this metadata file. The corresponding image contents are held by DVC and can be uploaded to or restored from the configured DagsHub remote.
