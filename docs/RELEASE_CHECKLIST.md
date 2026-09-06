# GitHub release checklist

- [x] Select and add an explicit source-code `LICENSE` (MIT).
- [x] Add machine-readable software citation metadata in `CITATION.cff`.
- [ ] Add the public paper/preprint URL and final BibTeX record to `README.md`.
- [ ] Upload large permitted artifacts outside ordinary Git and add stable download links.
- [ ] Confirm dataset redistribution terms before publishing processed JSON files.
- [x] Add a clean-environment GitHub Actions test workflow.
- [x] Run the curated test suite locally (138 passed, 3 external-artifact skips;
  6 September 2026).
- [ ] Confirm the GitHub Actions workflow passes on the public commit.
- [ ] Run all `SyntheticDryRun` and `Preflight` entry points from a clean clone.
- [x] Search tracked source and documentation for usernames, absolute local
  paths, credentials and tokens (6 September 2026).
- [x] Remove hard-coded Git executable paths from provenance helpers.
- [x] Confirm no file tracked by Git exceeds GitHub's 100 MB hard limit.
- [ ] Tag the exact commit used for the paper results.
- [ ] Archive the tagged release and record its DOI if available.
