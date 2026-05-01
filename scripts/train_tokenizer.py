from LaughLM.data.tokenizer_train import train_tokenizer
import shutil
import os
import time

start = time.time()

local_path = "/tmp/tokenizer.json"
drive_path = "/content/drive/MyDrive/LaughLM/tokenizer/tokenizer.json"

train_tokenizer(
    dataset_name  = "dignity045/tokenizer_dataset_v1",
    vocab_size    = 32000,
    max_samples   = 1_000_000,
    num_workers   = 16,
    work_dir      = "/tmp/tok_work",
    output_path   = local_path,
)

# Copy to Google Drive after training
os.makedirs(os.path.dirname(drive_path), exist_ok=True)
shutil.copy(local_path, drive_path)

print("✓ Tokenizer copied to Drive")
print("Total time:", (time.time()-start)/60, "minutes")
