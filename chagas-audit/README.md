# chagas-audit — reader-facing web app

Single-page static web app used to collect independent re-audit scores from
three readers on 84 quadrant images from the Morais *et al.* (2022) *T. cruzi*
blood-smear dataset. Produced for the PLOS Neglected Tropical Diseases
manuscript on FPN-based image-level classification of *T. cruzi*.

## What it is
- A single `index.html` (vanilla JS, no build step) plus 84 JPEG images.
- Each reader visits the deployed URL with their own query parameter
  (`?reader=lidia`, `?reader=katie`, `?reader=chris`) so their scores are
  tagged with their name.
- A password gate prevents casual access; the shared password is sent to
  readers by email separately.
- Progress is auto-saved in `localStorage` after every click — readers can
  close the tab and resume any time in the same browser.
- On completion, the reader exports a JSON file and emails it to Chris.

## Password
- Default: **`chagas2026`**.
- To change it, compute `sha256` of your new password and update the
  `PW_HASH_HEX` constant near the top of the `<script>` block in
  `index.html`. The page does not store the password, only its hash.

## Deployment — GitHub Pages

1. Create a private repository on GitHub (e.g. `chagas-audit`).
2. Push this directory:
   ```bash
   cd chagas-audit
   git init -b main
   git add .
   git commit -m "Initial audit app"
   git remote add origin git@github.com:<user>/chagas-audit.git
   git push -u origin main
   ```
3. In the GitHub web UI for the repo: **Settings → Pages →**
   **Source: Deploy from a branch**, **Branch: `main` / root**.
4. Wait ~30 s for GitHub to build and serve. The URL will be
   `https://<user>.github.io/chagas-audit/`.
5. Confirm the URL loads, shows the password gate, and works end-to-end
   with one test score before you send it to readers.

## Sending URLs to readers
Send each reader a personalized URL with their name in the query string:

- `https://<user>.github.io/chagas-audit/?reader=lidia`
- `https://<user>.github.io/chagas-audit/?reader=katie`
- `https://<user>.github.io/chagas-audit/?reader=chris`

Include the password (`chagas2026` by default) in the same email.

## When scores return
Each reader will email you a file named `scores_<name>_<timestamp>.json`.
Save these to `../audit_responses/` in the main chagas project and run the
analysis script (to be written after scores arrive) to join them with
`audit_package/_mapping_DO_NOT_SHARE.csv` and compute the aggregate stats.
