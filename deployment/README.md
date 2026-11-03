# GitHub Pages deployment

The project site is [jamesliuai.github.io/osu-lab](https://jamesliuai.github.io/osu-lab/). Its project base path is `/osu-lab/`. GitHub Pages uses **GitHub Actions** as its publishing source.

The [Pages workflow](../.github/workflows/pages.yml) runs only through `workflow_dispatch` in `jamesliuai/osu-lab`. It checks `tools/verify_publication.py --site` before packaging **only `site/`**. The build job has read access to repository contents. The deploy job needs the build, uses the `github-pages` environment and has `pages: write` and `id-token: write` permissions. Deployments share one concurrency group and do not cancel a running deployment.

## Publish a verified revision

1. Run `python -I -S -B tools/verify_publication.py --site` on a clean checkout. Reconcile the inventory, file provenance, rights scopes and checksums whenever publication files change. Run the software's strict package, claim, study, document, figure and license checks if their inputs change.
2. Verify that the four versioned [release downloads](../release/DOWNLOAD_MANIFEST.json) are publicly available and match [their checksums](../release/SHA256SUMS). Version `v0.1.0rc6` is a prerelease. Do not rebuild or substitute its prepared assets.
3. In repository **Settings → Pages**, select **GitHub Actions**. Ensure Actions is enabled and the `github-pages` environment permits the intended default-branch deployment.
4. In **Actions → Publish verified osu-lab site → Run workflow**, select `main`. The equivalent CLI command is `gh workflow run pages.yml --repo jamesliuai/osu-lab --ref main`.
5. Confirm both jobs succeed, then open the site without authentication. Check the `/osu-lab/` base path, assets, navigation, Summary/Report/Code links, desktop/mobile layout and console. Fetch all release downloads and compare their SHA-256 values with the published metadata.

The website uses relative assets and versioned Release URLs. The nine-file Pages artifact contains no software/evidence tree, private handoff, local build tooling, preview records or download copies. Never publish the repository root as a Pages artifact.

## Local preview

Serve `site/` through a local HTTP server, for example `python -m http.server 8766 --directory site`, then open `http://localhost:8766/`. This previews the content at `/`; deployment verification must also test the `/osu-lab/` project path.

The software and prepared archives preserve their original preparation records byte-for-byte. Those records describe checks performed before this repository's publication; this deployment guide describes the operational workflow. No original research Git history is included.

Workflow reference: [GitHub's custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
