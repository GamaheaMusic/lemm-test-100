"""
Force upload app.py to HuggingFace Space
"""
import os
from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN')
if not token:
    print("❌ HF_TOKEN not set!")
    exit(1)

api = HfApi(token=token)
space_id = "Gamahea/lemm-test-100"

print(f"📤 Force uploading app.py to {space_id}...")

# Upload with unique commit message to force update
import time
api.upload_file(
    repo_id=space_id,
    repo_type="space",
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    commit_message=f"Fix Gradio schema error: use None for timeline_state default ({int(time.time())})"
)

print("✅ Upload complete! Space will restart automatically.")
print(f"🔗 View at: https://huggingface.co/spaces/{space_id}")
