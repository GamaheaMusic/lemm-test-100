"""
Upload Phase 1 implementation to HuggingFace Space v1.0.2 branch
"""
from huggingface_hub import HfApi
import os

# Initialize API
api = HfApi()

# Space details  
repo_id = "Gamahea/lemm-test-100"
repo_type = "space"
branch = "v1.0.2"

print(f"Uploading Phase 1 files to branch '{branch}'...")

# Phase 1 files to upload
files_to_upload = [
    # Core application
    "app.py",
    "version.py",
    "requirements.txt",
    "requirements_hf.txt",
    
    # New MSD services
    "backend/services/msd_database_service.py",
    "backend/services/genre_profiler.py",
    "backend/services/msd_importer.py",
    
    # Documentation
    "PHASE1_IMPLEMENTATION_COMPLETE.md",
    "IMPLEMENTATION_PLAN_v1.0.2.md",
    "MILLION_SONG_DATASET_INTEGRATION_REPORT.md",
    
    # Test script
    "test_msd_services.py"
]

print(f"\nUploading {len(files_to_upload)} files to branch '{branch}'...")

uploaded = 0
failed = 0

for file_path in files_to_upload:
    if os.path.exists(file_path):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=branch,
                commit_message=f"Phase 1: Update {file_path}"
            )
            print(f"✅ Uploaded {file_path}")
            uploaded += 1
        except Exception as e:
            print(f"❌ Failed to upload {file_path}: {e}")
            failed += 1
    else:
        print(f"⚠️ File not found: {file_path}")
        failed += 1

print(f"\n{'='*60}")
print(f"Upload Summary:")
print(f"✅ Uploaded: {uploaded}")
print(f"❌ Failed: {failed}")
print(f"{'='*60}")

if uploaded > 0:
    print(f"\n🎉 Phase 1 files uploaded successfully!")
    print(f"🔗 View at: https://huggingface.co/spaces/{repo_id}/tree/{branch}")
else:
    print("\n⚠️ No files were uploaded. Check errors above.")
