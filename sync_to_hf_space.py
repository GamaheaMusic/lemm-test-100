"""
Sync updated files to HuggingFace Space
"""
import os
from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN')
if not token:
    print("❌ HF_TOKEN not set!")
    exit(1)

api = HfApi(token=token)
space_id = "Gamahea/lemm-test-100"

print(f"📤 Syncing files to {space_id}...")

# Upload specific updated files
files_to_sync = [
    ("app.py", "app.py"),
    ("backend/services/diffrhythm_service.py", "backend/services/diffrhythm_service.py"),
]

for local_path, repo_path in files_to_sync:
    print(f"   Uploading {local_path}...")
    api.upload_file(
        repo_id=space_id,
        repo_type="space",
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        commit_message=f"Add LoRA support to music generation - load from HF and local sources"
    )

print("✅ Sync complete! Space will restart automatically.")
print(f"🔗 View at: https://huggingface.co/spaces/{space_id}")
