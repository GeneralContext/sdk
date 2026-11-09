# Releasing the SDKs

Both SDKs are Apache-2.0 and publish from `.github/workflows/release.yml` when a tag is
pushed. Nothing is uploaded from a laptop.

| Package | Registry | Tag |
| --- | --- | --- |
| `generalcontext` (python/) | PyPI | `python-vX.Y.Z` |
| `@generalcontext/sdk` (node/) | npm | `node-vX.Y.Z` |

## One-time setup

1. **GitHub environment.** In the repository's settings, create an environment named
   `sdk-release` and add required reviewers, so a release needs a second person's approval.
2. **PyPI.** Sign in to a company PyPI account (2FA on). Under "Publishing", add a pending
   trusted publisher: project `generalcontext`, owner `GeneralContext`, repository
   `sdk`, workflow `release.yml`, environment `sdk-release`. No token is stored.
3. **npm.** Confirm the `generalcontext` org on npm belongs to us (or create it). Create a
   granular automation token limited to `@generalcontext/sdk` and save it as the `NPM_TOKEN`
   secret on the `sdk-release` environment. Once the package exists, switch it to npm trusted
   publishing and delete the token.
4. **Rehearse** (optional): point the Python job at TestPyPI by adding
   `repository-url: https://test.pypi.org/legacy/` to the publish step and tagging a `.dev` version.

## Each release

1. Bump the version in `python/pyproject.toml` and `__version__` in
   `python/generalcontext/__init__.py`, or in `node/package.json` and `VERSION` in
   `node/src/index.ts`.
2. Merge to `main`.
3. Tag and push: `git tag python-v0.2.0 && git push origin python-v0.2.0` (or `node-v0.2.0`).
4. Approve the `sdk-release` deployment. The job checks the tag matches the version, runs the
   tests, builds, and publishes.
5. Check it installs: `pip install generalcontext==0.2.0` or `npm install @generalcontext/sdk@0.2.0`.

A published version can't be replaced. To fix one, release the next patch version.

`LICENSE` and `NOTICE` live at the repository root; each package carries an identical copy
(CI checks they match), because a published package must include them.
