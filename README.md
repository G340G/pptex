# VHS PPT ARG Generator

Generates a creepy 90s-PPT / VHS found-tape job training horror video (MP4 with audio).

## Quick start (GitHub-only)

1) Add your images:
- `assets/local/` (objects, rooms, random photos)
- `assets/faces/` (faces/portraits)

2) Go to **Actions → Generate VHS PPT ARG Video → Run workflow**
- Set `seed`, `intensity`, `keywords`, etc.
- Run it

3) Download artifact:
- Open the workflow run → Artifacts → `vhs-ppt-arg-video` → `out.mp4`

## Optional: config.yaml
Copy `config.example.yaml` to `config.yaml` and edit.

## Notes
- Web scraping uses Wikimedia Commons API (no keys). You are responsible for attribution/licensing if you publish.
- For maximum uncanny: add more faces and office-ish training imagery.
