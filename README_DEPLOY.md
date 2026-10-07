# Deploying to Hugging Face Spaces (free tier)

I can't push to Hugging Face directly from here (no network access to
huggingface.co in this sandbox), so this is the "do it yourself" step —
but everything needed on your end is ready in this folder.

## What's already prepared for you
- `app.py` — the Streamlit app (tested locally, confirmed working)
- `predict_demo.py` and `pii_redaction.py` — core pipeline + PII masking,
  both imported directly by `app.py` (the app will crash on import if
  `pii_redaction.py` isn't pushed alongside it)
- `requirements.txt` — trimmed to exactly what `app.py` needs
  (streamlit, pandas, scikit-learn, joblib)
- `models/*.pkl` — your trained models (vectorizer, priority, department,
  and `department_explain_coef.pkl` for the explainability panel), loaded
  directly by `app.py` via `predict_demo.py`

## Steps

1. **Create the Space**
   Go to https://huggingface.co/new-space
   - Owner: your account
   - Space name: e.g. `support-ticket-tagger`
   - License: whatever you prefer (MIT is fine for a student project)
   - **Space SDK: choose "Streamlit"**
   - Hardware: "CPU basic - free"
   - Click "Create Space"

2. **Add the required front-matter to README.md**
   Hugging Face Spaces reads a small YAML header at the top of `README.md`
   to know how to build the Space. Add this to the very top of your
   project's `README.md` (create one if you don't have it yet):

   ```yaml
   ---
   title: Support Ticket Auto-Tagger
   emoji: 🎫
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   sdk_version: "1.38.0"
   app_file: app.py
   pinned: false
   ---
   ```

   (Run `streamlit --version` locally to confirm the exact version to put
   in `sdk_version` — it should match what's in your `requirements.txt`
   environment.)

3. **Push your files**
   Hugging Face Spaces are just git repos. From your project folder:

   ```bash
   git init
   git remote add space https://huggingface.co/spaces/<your-username>/support-ticket-tagger
   git add app.py predict_demo.py pii_redaction.py requirements.txt models/ README.md
   git commit -m "Initial deploy"
   git push space main
   ```

   (You'll be prompted for your HF username + an access token — generate
   one at https://huggingface.co/settings/tokens with "write" access.)

4. **Wait for the build**
   The Space will show a build log; once it says "Running", your app is
   live at `https://huggingface.co/spaces/<your-username>/support-ticket-tagger`.

## Gotchas to watch for
- **Only commit `models/*.pkl`, not `data/raw/customer_support_tickets.csv`**
  if you'd rather not publish the raw Kaggle data publicly — the app only
  needs the trained `.pkl` files, not the CSV, to run.
- If the Space build fails on `sklearn` version mismatch warnings, that's
  usually harmless (same warning you may have seen locally) — it's only a
  problem if predictions look wrong, not just because the warning appears.
- Free-tier Spaces sleep after inactivity and take ~30s to wake up on the
  next visit — normal, not a bug.
