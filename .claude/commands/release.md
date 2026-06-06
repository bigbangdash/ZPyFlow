Perform a ZPyFlow release. The argument is the new version string (e.g. `0.1.2`).

## Steps

**1. Verify current branch**
Confirm we are on `develop`. If not, stop and tell the user.

**2. Bump version**
Update the version string in both files (replace the old `X.Y.Z` with the new one):
- `Cargo.toml` — `version = "X.Y.Z"` in `[package]`
- `pyproject.toml` — `version = "X.Y.Z"` in `[project]`

**3. Update CHANGELOG.md**
- Change `## [Unreleased]` to `## [Unreleased]` (keep the section, but it should be empty)
- Add a new `## [NEW_VERSION] — YYYY-MM-DD` section above the previous release, summarising what changed since the last tag. Use `git log vPREV...HEAD --oneline` to enumerate commits.
- Update the footer links:
  - `[Unreleased]` diff link: `.../compare/vNEW...HEAD`
  - Add `[NEW_VERSION]` diff link: `.../compare/vPREV...vNEW`

**4. Commit and push to develop**
```
git add Cargo.toml pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to vNEW_VERSION"
git push origin develop
```

**5. Create PR to main**
```
gh pr create --base main --head develop \
  --title "release: v<VERSION>" \
  --body "Release v<VERSION> — see CHANGELOG.md for details."
```
Print the PR URL and **stop here**. Tell the user to review and merge the PR, then run `/smoke-test <VERSION>` after merge and tag.

---

**Post-merge steps (run after user confirms PR is merged):**

**6. Switch to main and pull**
```
git checkout main && git pull origin main
```

**7. Create and push the tag**
```
git tag v<VERSION>
git push origin v<VERSION>
```

**8. Create GitHub Release**
```
gh release create v<VERSION> \
  --title "v<VERSION>" \
  --notes "See [CHANGELOG.md](CHANGELOG.md) for details." \
  --latest
```
This triggers `release.yml` which builds wheels and publishes to PyPI.

Print the release URL and tell the user to run `/smoke-test <VERSION>` once PyPI has the new wheel (usually 2–5 minutes).
