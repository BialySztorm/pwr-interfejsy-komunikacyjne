# How to get a Google Drive API key

1. Go to https://console.cloud.google.com and sign in with a Google account.
2. Create a new project (project selector at the top → **New Project**, e.g. name it
   `sentiment-benchmark`).
3. **APIs & Services → Library** → search "Google Drive API" → **Enable**.
4. **APIs & Services → Credentials → Create credentials → API key**; copy the generated key.
5. Recommended: **Edit API key → API restrictions → Restrict key** → check only
   "Google Drive API" → **Save** (so the key is useless for anything else if it leaks).
6. Paste the key into `.env` as `GDRIVE_API_KEY=...`, and the folder link as
   `GDRIVE_FOLDER_URL=...`.
7. Run `python scripts/sync_drive.py --dry-run` to check the connection.

## Notes

- No billing account is needed — reading a public folder stays well within the free quota.
- Enabling the API (step 3) can take a few minutes to propagate. If you get a 403 right after
  enabling it, wait a bit and retry.
- Don't commit the key to the repository or paste it into a team chat/channel — each team
  member should generate their own key in their own GCP project.
