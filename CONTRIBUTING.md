# Contributing to hqnn-forge

Thanks for your interest in contributing. This document summarizes the workflow we follow so
changes stay easy to review and the history stays easy to read.

Participation in issues, pull requests, and reviews is covered by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Language

All project communication must be in **English** — README, docs, commit messages, PR
titles/descriptions, code comments, issue titles and comments. Keeping everything in one
language ensures the history is readable by any contributor or reviewer, regardless of
background.

## Guiding Principle: Keep it Clean and Simple

Reduce complexity and increase clarity. A clean `main` branch, understandable commit history,
and consistent processes lead to better software.

Adding a dataset loader, an encoding layer or a variational block? The conventions and required
tests for each are in [`docs/extending.md`](docs/extending.md).

## Git Workflow

### 1. Issue First

Every change should start with a GitHub issue before a branch is opened. Issues provide context
for why a change is being made, create a reference point for discussion, and ensure the work is
intentional.

*   **Title Format:** Issue titles use the same conventional-commit prefixes as branches and
    commits (e.g., `feat:`, `fix:`, `docs:`).
*   **Description Structure:** Provide clear context, steps to reproduce (for bugs), and
    expected outcomes (for features).
*   **Exception:** Trivial fixes (typos, obvious broken links) may skip the issue and go
    directly to a branch and PR.
*   **Order doesn't matter:** Issues can be worked, edited, or commented on in whatever order
    the work actually requires — there's no need to process them in creation order. The only
    real constraint is the reverse: don't reference an issue number in a commit, PR, or another
    issue before that issue exists.

### 2. Branch Naming

Feature branches are named `<type>/<short-description>`, using the same type prefix as the
conventional commit that will result from the PR (e.g. `feat/user-auth`, `fix/login-bug`,
`docs/api-guide`). Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

### 3. No Direct Commits to Main

All changes go through a feature branch and a pull request — never commit directly to `main`.
This keeps the history reviewable and associates every change with a PR number.

### 4. Pull Requests

*   **One PR per feature branch**, addressing a single, specific purpose.
*   **PR title** follows the same conventional-commit format as the resulting squash-merge
    commit (e.g. `feat: add user authentication`).
*   **Test plan required** for any PR that changes code behaviour: a markdown checklist of
    steps to verify the change works, checked off before merging. Pure documentation or
    configuration changes don't need one.
*   **Stacked PRs.** A PR may use another feature branch as its base, when it builds on
    work not yet on `main`. CI runs on it when it is opened, when its own branch is pushed
    and when it is retargeted, but **not when its base branch is pushed**, so a review fix
    on the parent can break a child whose check stays green. After pushing to a branch
    other PRs are based on, re-run them by closing and reopening each:

    ```bash
    for n in $(gh pr list --base <branch> --json number --jq '.[].number'); do
        gh pr close "$n" && gh pr reopen "$n"
    done
    ```

    Reopening fires a fresh run against the current merge result. Re-running the old
    run from the Actions tab does not help: it checks out the merge commit it was
    created with. This isn't automated because events caused by the workflow's own
    `GITHUB_TOKEN` start no new workflow runs, so it would need a personal access token
    or GitHub App secret. Before squash-merging the parent, retarget the children to
    `main` (`gh pr edit <n> --base main`): the repository deletes a head branch as soon as
    its PR merges, and depending on how the branch is deleted, GitHub either retargets
    the PRs based on it or closes them.

### 5. Squash Merge

PRs targeting `main` are squash-merged, condensing the branch's history into one commit whose
title includes both the issue number and the PR number (e.g. `feat: add user authentication
(fixes #12) (#42)`).

Experiment/research branches are the exception — use a regular merge there so the trail of what
was tried and why isn't lost to squashing.

### 6. Review

Most PRs are self-reviewed before merging: does the code work as intended, does it follow
project conventions, are there obvious security issues, is test coverage adequate, is
documentation updated. Changes that modify interfaces or break existing contracts benefit from
review by affected stakeholders before merging.

How a PR reaches review:

*   **The first request is automatic.** `.github/CODEOWNERS` requests a review from
    @g8rdier on every PR, including PRs from forks: when it is opened, or for a draft, when
    it is marked ready for review.
*   **After pushing changes that address review comments**, say so in a PR comment that
    mentions @g8rdier (e.g. "@g8rdier ready for another look"). From a fork, GitHub doesn't
    let the author re-request a review in the sidebar, so the mention is the hand-off.
    Collaborators with write access use "Re-request review" in the sidebar instead.
*   **On fork PRs, keep "Allow edits by maintainers" enabled**, so small fixes can be pushed
    to the branch directly instead of going back and forth in comments.
*   CI on a first-time contributor's fork PR waits for a maintainer to approve the workflow
    run; that's expected and needs nothing from the contributor.

### 7. Hotfixes

Critical production issues use a `hotfix/` branch. An issue is still recommended (can be
created retroactively) and a PR is still required, but review can be expedited with a
stakeholder's immediate approval. Direct commits to `main` remain forbidden.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Commit titles
are entirely lowercase, prefixed with a type: `feat`, `fix`, `docs`, `style`, `refactor`,
`test`, `chore`.

```
feat: add user authentication endpoint
```

Keep commit bodies to at most 3 bullet points. If you need more, the work is probably better
split into smaller, more atomic commits.

## Testing

CI runs the whole suite. For the local edit–test loop, leave out the tests marked `slow`
(end-to-end training, the gradient-variance physics checks, parameter-shift batching, repeated
fits and bootstraps), which take most of its run time:

```bash
uv run --frozen --all-extras pytest -m "not slow"   # about a minute
uv run --frozen --all-extras pytest                 # the full suite, as CI runs it
```

Mark a test `@pytest.mark.slow` when it takes 1.2 s or more (`pytest --durations=50` finds
them); `tests/conftest.py` records the rule. `--strict-markers` is on, so a misspelled marker
fails collection instead of silently leaving a slow test in the quick run.

## Linting

The `lint` job in `.github/workflows/tests.yml` runs four checks, and a PR must pass all of
them. To run the same checks locally with the same tool versions, run them through the
lockfile:

```bash
uv run --frozen --all-extras ruff check .
uv run --frozen --all-extras ruff format --check .   # `ruff format .` applies the fixes
uv run --frozen --all-extras mypy hqnn_forge tests examples
uv run --frozen --all-extras vermin --no-tips -t=3.11- --violations --eval-annotations \
    --exclude long hqnn_forge tests examples .github/scripts
```

*   **ruff** lints the whole tree and checks its formatting, including the Python code
    blocks in Markdown files. The pre-commit hooks in the README run both steps on each
    commit.
*   **mypy** type-checks the package, the tests and the examples. Modules that still fail
    are listed under `[[tool.mypy.overrides]]` in `pyproject.toml` until #232 clears them.
    CI also fails when an entry there no longer matches any module.
*   **vermin** fails on stdlib calls that don't exist on Python 3.11, the `requires-python`
    floor, which neither ruff nor mypy catches. CI also fails when vermin reports a file as
    "incompatible", because vermin skips such a file without an error.
*   **Why `uv run --frozen`.** It uses the ruff, mypy and vermin versions pinned in
    `uv.lock`, which CI uses too. `--all-extras` installs every extra, as CI does. Without it, mypy
    types matplotlib, the one typed package among the extras, as `Any`, because
    `ignore_missing_imports` hides the missing module. The `dev` extra declares the tools without a version range, so a pip install
    gets the newest releases, and when a new ruff changes a default, its `ruff format`
    disagrees with CI's. The versions are deliberately not capped in `pyproject.toml`
    either. Dependabot only updates `uv.lock`, so a cap such as `ruff<0.17` would stop it
    from ever proposing the next minor release.

## Dependency Changes

`uv.lock` is tracked in the repository and CI tests against it: the `test-locked` job in
`.github/workflows/tests.yml` runs `uv sync --locked --all-extras`, which installs the exact
versions the lockfile records and fails if the lockfile no longer matches `pyproject.toml`.

Installing the package doesn't need uv; the README documents a pip install for that. The
README's development setup uses uv, and uv is needed to regenerate the lockfile; see the [uv
installation docs](https://docs.astral.sh/uv/getting-started/installation/) if you don't have
it.

*   **After editing `pyproject.toml`**, regenerate the lockfile and commit it in the same PR:

    ```bash
    uv lock
    git add pyproject.toml uv.lock
    ```

    This applies to every hand edit of a field the lockfile records: adding or removing a
    package, changing a version floor, touching an extra — and also the project's own
    `version` and `requires-python`, which `uv.lock` pins as well. A release PR that only
    bumps `version` still needs `uv lock`.

*   **What the failure looks like** when the lockfile is stale:

    ```
    error: The lockfile at `uv.lock` needs to be updated, but `--locked` was provided.

    hint: To update the lockfile, run `uv lock`.
    ```

    The fix is exactly that hint: run `uv lock` locally and push the updated `uv.lock`.

*   **Dependabot bumps the lockfile on its own.** Its PRs update `uv.lock` without touching
    `pyproject.toml`, so contributors only regenerate the lockfile when they change
    `pyproject.toml` by hand.

*   **Upcoming PennyLane releases are tested weekly.** `.github/workflows/upstream.yml` runs
    the suite against the newest PennyLane and pennylane-lightning pre-releases on PyPI and
    against their nightly builds on TestPyPI. It never blocks a PR. When it fails, it opens
    (or comments on) an issue titled `ci: test suite fails against PennyLane <source> builds`:
    the library relies on PennyLane internals, and this is how a break shows up before users
    upgrade. Trigger it by hand with `gh workflow run upstream.yml`.

## Documentation

The API reference site is built from the docstrings by `mkdocs build --strict`
(`mkdocs.yml`, `docs/`), which the Docs workflow runs on every PR. Docstrings are NumPy style
and cross-reference with Sphinx roles (`` :func:`gradient_variance` ``,
`` :class:`~hqnn_forge.models.HybridBinaryClassifier` ``); `docs/griffe_sphinx_roles.py` turns
them into links, so a reference to something that does not exist fails the build. Constants are
documented with `#:` comments above the assignment. A new public module needs a page under
`docs/api/` and an entry in the `nav` of `mkdocs.yml`; the workflow fails if a top-level
module has no page or a name in a documented `__all__` has no entry.

## Versioning

Releases follow [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`), tagged (e.g.
`v1.2.0`) on the `main` merge commit that encapsulates the release. `CHANGELOG.md` follows
[Keep a Changelog](https://keepachangelog.com/).

*   **Every user-facing PR adds its changelog line** under `## [Unreleased]`, in `Added`,
    `Changed`, `Fixed` or `Removed`, ending with the PR number, e.g. `(#42)`. User-facing means
    anything a user of the package can observe: the API, behaviour, results, dependencies,
    supported Python versions. CI, tests, internal refactors and contributor docs don't need
    one. A test checks that every entry names its PR.
*   **A release PR** sets `version` in `pyproject.toml` (then `uv lock`), renames
    `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and opens a new empty `## [Unreleased]`
    above it, and adds the compare link at the bottom. It also bumps `version` in
    `CITATION.cff` to the new version (`tests/test_citation.py` fails until it does) and can
    add a `date-released`.
*   **Tagging publishes.** After the release PR is merged, tag its merge commit and push the
    tag:

    ```bash
    git tag vX.Y.Z <merge-commit> && git push origin vX.Y.Z
    ```

    `.github/workflows/release.yml` then checks that the tag equals `v` + `project.version`,
    that the tagged commit is on `main`, and that the changelog has a non-empty `[X.Y.Z]`
    section (`.github/scripts/release_notes.py`). It builds and checks the sdist and wheel and
    runs the locked test suite on that commit. Only then does the publish job, in the `pypi`
    environment, wait for the owner's approval, publish to PyPI through trusted publishing (no
    stored token), and create the GitHub release with that changelog section as its notes. A
    manual run of the workflow on `main` is a dry run to TestPyPI; it refuses any other branch.
*   **The `pypi` environment must be protected before the first tag**: required reviewer the
    owner, deployments limited to `v*` tags. A job that names an environment the repository
    does not have creates it unprotected, so without this setup a pushed tag publishes with
    no approval step.

## Using AI Coding Assistants

If you use an AI coding assistant to help write a contribution, that's fine on your feature
branch — but review and understand everything before opening the PR, the same as you would for
your own code: check it follows project conventions, look for security issues, confirm tests
are adequate and passing. You are responsible for what you submit regardless of how it was
produced. Commits should carry human authorship only — please don't include AI co-authorship
trailers (e.g. `Co-Authored-By: <assistant>`) or tool-generated session links in commits, PR
descriptions, or comments in this repository.
