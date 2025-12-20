"""
Upload v1.0.2 branch to HuggingFace Space
Creates a new branch without binary file history
"""
from huggingface_hub import HfApi
import os

# Initialize API
api = HfApi()

# Space details  
repo_id = "Gamahea/lemm-test-100"
repo_type = "space"
branch = "v1.0.2"

# Create the branch first (from main)
print(f"Creating branch '{branch}' from main...")
try:
    api.create_branch(repo_id=repo_id, repo_type=repo_type, branch=branch, revision="main")
    print(f"✅ Created branch {branch}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"ℹ️ Branch {branch} already exists")
    else:
        print(f"⚠️ Error creating branch: {e}")

# Upload the modified files to the new branch
files_to_upload = [
    "app.py",
    "version.py",
    "README.md",
    "MILLION_SONG_DATASET_INTEGRATION_REPORT.md",
    ".gitignore",
    "requirements_hf.txt",
    "backend/services/audio_analysis_service.py",
    "backend/services/diffrhythm_service.py",
    "backend/services/export_service.py",
    "backend/services/hf_storage_service.py"
]

print(f"\nUploading {len(files_to_upload)} files to branch '{branch}'...")

for file_path in files_to_upload:
    if os.path.exists(file_path):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=branch,
                commit_message=f"v1.0.2: Update {file_path}"
            )
            print(f"✅ Uploaded {file_path}")
        except Exception as e:
            print(f"❌ Failed to upload {file_path}: {e}")
    else:
        print(f"⚠️ File not found: {file_path}")

print(f"\n🎉 Branch '{branch}' created and updated successfully!")
print(f"🔗 View at: https://huggingface.co/spaces/{repo_id}/tree/{branch}")
