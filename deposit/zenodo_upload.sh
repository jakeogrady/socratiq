#!/usr/bin/env bash
# Upload files into the saved Zenodo draft through the Zenodo API, then check
# each file's MD5 against the local copy.
#
# Usage (from the repository root):
#   ./deposit/zenodo_upload.sh deposit/zenodo-v2/socratiq-v2-adapters-qwen3.zip deposit/zenodo-v2/socratiq-v2-adapters-llama3.2.zip
#
# The script asks for a Zenodo personal access token and never prints or stores it.
# Create one at: Zenodo -> your name (top right) -> Applications -> Personal access
# tokens -> New token, with the scopes deposit:write and deposit:actions.
set -euo pipefail

RECORD_ID="23236018"   # from the reserved DOI 10.5281/zenodo.23236018
API="https://zenodo.org/api/deposit/depositions/${RECORD_ID}"

if [ "$#" -eq 0 ]; then
  echo "Give the files to upload." >&2
  exit 1
fi
for file in "$@"; do
  [ -f "$file" ] || { echo "Not found: $file" >&2; exit 1; }
done

read -r -s -p "Zenodo personal access token (input hidden): " TOKEN
echo
AUTH="Authorization: Bearer ${TOKEN}"

# The draft's file bucket.
BUCKET=$(curl -sS -f -H "$AUTH" "$API" | python3 -c 'import json,sys; print(json.load(sys.stdin)["links"]["bucket"])') || {
  echo "Could not open draft ${RECORD_ID}. Check the token scopes and that the draft is saved." >&2
  exit 1
}
echo "Draft bucket found."

for file in "$@"; do
  name=$(basename "$file")
  echo "Uploading ${name} ($(du -h "$file" | cut -f1))..."
  reply=$(curl -f --progress-bar -H "$AUTH" --upload-file "$file" "${BUCKET}/${name}")
  remote=$(printf '%s' "$reply" | python3 -c 'import json,sys; print(json.load(sys.stdin)["checksum"].removeprefix("md5:"))')
  local_md5=$(md5 -q "$file")
  if [ "$remote" = "$local_md5" ]; then
    echo "  OK: ${name} uploaded, MD5 ${remote} matches."
  else
    echo "  MISMATCH: ${name} Zenodo MD5 ${remote}, local ${local_md5}. Upload it again." >&2
  fi
done
unset TOKEN AUTH
echo "Done. Reload the draft page in Zenodo to see the files."
