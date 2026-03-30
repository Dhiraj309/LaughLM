from LaughLM.data.tokenizer_train import train_tokenizer  
import time  
# import os  
  
# os.environ["RAYON_RS_NUM_CPUS"] = "4"  
# os.environ["TOKENIZERS_PARALLELISM"] = "true"  
  
start_time= time.time()  
  
train_tokenizer(  
    dataset_name = "dignity045/tokenizer_dataset_v1",  
    vocab_size = 32_000,  
    output_path = "tokenizer/tokenizer.json",  
    min_frequency = 2,  
    # shuffle_buffer = 1,  
    batch_size = 10_000,  
    max_samples = 10_000,  
)  
  
end_time = time.time()  
elapsed_time = end_time - start_time  
  
print(f"\n⏱️ Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
