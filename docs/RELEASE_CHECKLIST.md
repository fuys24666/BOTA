# GitHub release checklist

- [ ] Select and add an explicit source-code `LICENSE`.
- [ ] Add the public paper/preprint URL and final BibTeX record to `README.md`.
- [ ] Upload large permitted artifacts outside ordinary Git and add stable download links.
- [ ] Confirm dataset redistribution terms before publishing processed JSON files.
- [ ] Run `pytest -q` in a fresh environment.
- [ ] Run all `SyntheticDryRun` and `Preflight` entry points from a clean clone.
- [ ] Search the repository for usernames, absolute local paths, credentials and tokens.
- [ ] Confirm no file tracked by Git exceeds GitHub's 100 MB hard limit.
- [ ] Tag the exact commit used for the paper results.
- [ ] Archive the tagged release and record its DOI if available.
