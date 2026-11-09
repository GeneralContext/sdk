# Releasing the SDKs

Both SDKs are Apache-2.0 and publish from `.github/workflows/release.yml` when a tag is
pushed. Nothing is uploaded from a laptop.

| Package | Registry | Tag |
| --- | --- | --- |
| `generalcontext` (python/) | PyPI | `python-vX.Y.Z` |
| `generalcontext` (node/) | npm | `node-vX.Y.Z` |

## One-time setup

1. **GitHub environment.** In the repository's settings, create an environment named
   `sdk-release`. Add required reviewers, turn on "Prevent self-review", and under
   "Deployment branches and tags" allow only tags matching `python-v*` and `node-v*`.
2. **Protect `main` and the tags.** Add a ruleset for `main` requiring a pull request with one
   approval and the `CI` checks, and blocking force pushes and deletion. Add a tag ruleset for
   `python-v*` and `node-v*` that only admins can create, update or delete.
3. **Private vulnerability reporting.** Settings, Code security: turn on private vulnerability
   reporting (available once the repository is public).
4. **PyPI.** Sign in to a company PyPI account (2FA on). Under "Publishing", add a pending
   trusted publisher: project `generalcontext`, owner `GeneralContext`, repository
   `sdk`, workflow `release.yml`, environment `sdk-release`. No token is stored.
5. **npm.** The package is the unscoped `generalcontext`, owned by the `generalcontext` npm org.
   Publish through npm trusted publishing (package settings > Trusted Publisher: GitHub Actions,
   org `GeneralContext`, repo `sdk`, workflow `release.yml`, environment `sdk-release`). Only
   the very first publish of a new package needs a short-lived granular token with bypass-2FA,
   stored as the `NPM_TOKEN` secret on the `sdk-release` environment and deleted afterwards.
6. **Rehearse** (optional): point the Python job at TestPyPI by adding
   `repository-url: https://test.pypi.org/legacy/` to the publish step and tagging a `.dev` version.

## Each release

1. Bump the version in `python/pyproject.toml` and `__version__` in
   `python/generalcontext/__init__.py`, or in `node/package.json` and `VERSION` in
   `node/src/index.ts`.
2. Merge to `main`.
3. Tag and push: `git tag python-v0.2.0 && git push origin python-v0.2.0` (or `node-v0.2.0`).
4. The build job checks the tag matches the version, runs the tests, builds once, and signs a
   build-provenance attestation for the files.
5. Approve the `sdk-release` deployment. The publish job uploads exactly those files.
6. Check it: `pip install generalcontext==0.2.0` or `npm install generalcontext@0.2.0`, then
   `gh attestation verify <downloaded file> --repo GeneralContext/sdk`.

A published version can't be replaced. To fix one, release the next patch version.

`LICENSE` and `NOTICE` live at the repository root; each package carries an identical copy
(CI checks they match), because a published package must include them.
