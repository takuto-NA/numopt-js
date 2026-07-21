# Publishing

Release runbook for `numopt-js` (already published on npm).

## Preconditions

- Working tree is clean on the commit you intend to release
- `package.json` `version` matches the git tag you will push (`vX.Y.Z`)
- CI is green on that commit

## Release steps

1. Bump version in `package.json` (and keep README / docs consistent if needed).
2. Commit the version bump.
3. Create and push an annotated tag:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

4. GitHub Actions workflow `.github/workflows/publish.yml` runs on `v*` tags:
   - installs dependencies
   - runs `npm test`
   - runs `npm run smoke:pack`
   - publishes with npm Trusted Publishing (`npm publish --provenance --access public`)

5. Confirm the package on npm:

```bash
npm view numopt-js version
```

## Manual publish (workflow_dispatch)

Use the `publish` workflow’s `workflow_dispatch` input `publish_ref` with a tag such as `v0.4.0`. The ref’s version must match `package.json`.

## Local verification before tagging

```bash
npm test
npm run smoke:pack
npm run docs
```

TypeDoc HTML is generated in CI and deployed by `.github/workflows/docs.yml` on `main`. Do not commit `docs/`.

## GitHub Pages (one-time)

API docs are served from GitHub Actions (not the legacy `main` `/docs` folder).

1. Repo **Settings → Pages → Build and deployment → Source**: choose **GitHub Actions**.
2. Merge this branch (or push `docs.yml` to `main`), then confirm the **Deploy docs** workflow succeeds.
3. Verify https://takuto-na.github.io/numopt-js/ shows current TypeDoc output.

If Pages is still on legacy `/docs`, deleting committed `docs/` will blank the site until the source is switched.
