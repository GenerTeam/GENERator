import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import hashlib
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import time
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference on Parquet")
    parser.add_argument(
        "--data_type",
        default="eukaryote",
        choices=["eukaryote", "bacteria", "others"],
        help="Data type (eukaryote, bacteria, or others)."
    )
    parser.add_argument(
        "--data_path",
        default="hf://datasets/GenerTeam/sequence-recovery",
        help="Download from https://huggingface.co/datasets/GenerTeam/sequence-recovery"
    )
    parser.add_argument(
        "--model_path",
        default="GenerTeam/GENERator-eukaryote-1.2b-base",
        help="Download from https://huggingface.co/GenerTeam/GENERator-eukaryote-1.2b-base",
    )
    parser.add_argument(
        "--output_dir",
        default="./sequence_recovery_results",
        help="Directory to save output files."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=6144,
        help="Maximum input sequence length (truncate left, keep rightmost)."
    )
    parser.add_argument(
        "--gen_len",
        type=int,
        default=5,
        help="Number of tokens to generate beyond the prompt."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference."
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 for faster inference."
    )
    return parser.parse_args()

class SuppressSpecialTokensLogitsProcessor:
    """Suppress all special tokens during generation by setting their logits to -inf."""
    
    def __init__(self, special_token_ids: list):
        self.special_token_ids = special_token_ids

    def __call__(self, input_ids, scores):
        for token_id in self.special_token_ids:
            scores[:, token_id] = -float('inf')
        return scores

def calculate_accuracy(predictions: List[str], labels: List[str], seq_length: int = 30) -> List[float]:
    """Calculate accuracy for each sequence."""
    accuracies = []
    for label, pred in zip(labels, predictions):
        same_count = sum(1 for i in range(min(len(label), len(pred), seq_length)) 
                        if label[i] == pred[i])
        accuracies.append(same_count / seq_length)
    return accuracies

def process_data_shard(shard_id, sequences_data, args, dtype):
    """Process a single data shard and return predictions with hash_index"""
    # Set GPU for current process
    torch.cuda.set_device(shard_id)
    device = f"cuda:{shard_id}"
    
    # Set data type
    dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    
    print(f"Shard {shard_id}: Loading model on GPU {shard_id}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        dtype=dtype
    ).to(device)
    
    tokenizer.padding_side = "left"
    
    # Set up logits processor
    special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens)
    logits_processor = LogitsProcessorList([
        SuppressSpecialTokensLogitsProcessor(special_token_ids)
    ])
    
    # Extract sequence data
    sequences_shard = [item['sequence'] for item in sequences_data]
    indices_shard = [item['hash_index'] for item in sequences_data]
    total_sequences = len(sequences_shard)
    
    print(f"Shard {shard_id}: Processing {total_sequences} sequences...")
    
    predictions = []
    
    with tqdm(total=total_sequences, desc=f"Shard {shard_id}", unit="seq") as pbar:
        for i in range(0, total_sequences, args.batch_size):
            batch_seqs = sequences_shard[i:i + args.batch_size]
            batch_indices = indices_shard[i:i + args.batch_size]
            
            # Truncate sequences
            truncated_seqs = [
                '<s>' + seq[-((min(len(seq), args.max_seq_len) // 6) * 6):] 
                for seq in batch_seqs
            ]
            
            # Tokenize
            inputs = tokenizer(
                truncated_seqs,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(device)
            
            # Generate
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=args.gen_len, 
                    pad_token_id=tokenizer.pad_token_id, 
                    do_sample=False,
                    logits_processor=logits_processor
                )
            
            # Decode predictions
            batch_preds = tokenizer.batch_decode(
                outputs[:, -args.gen_len:], 
                skip_special_tokens=True
            )
            
            # Save results with hash_index
            for pred, hash_index in zip(batch_preds, batch_indices):
                predictions.append({
                    "hash_index": hash_index,
                    "pred": pred
                })
            
            pbar.update(len(batch_seqs))
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return predictions

def process_checkpoint(
    args: argparse.Namespace,
    dtype: str
) -> Dict:
    """Process a single checkpoint with data sharding across GPUs and return results."""
    print("\n" + "=" * 80)
    print("🧬  DNA SEQUENCE RECOVERY PIPELINE  🧬")
    print("=" * 80 + "\n")
    print(f"Processing checkpoint {args.model_path.split('/')[-1]} with {dtype}...")
    
    # Load data
    print("Loading data...")
    data_path = f"{args.data_path}/{args.data_type}/test.parquet"
    df = pd.read_parquet(data_path)
    total_sequences = len(df)
    
    # Generate unique hash indices for each sequence
    print("Generating hash indices for sequences...")

    # Add hash_index column directly to DataFrame
    df['hash_index'] = df.apply(
        lambda row: hashlib.md5(f"{row['sequence']}_{row.name}".encode()).hexdigest()[:16], 
        axis=1
    )
    
    # Determine number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs with data sharding")
    
    # Split data into num_gpus shards
    shard_size = (total_sequences + num_gpus - 1) // num_gpus
    shards = []
    
    for i in range(num_gpus):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_sequences)
        if start_idx < total_sequences:
            shard_df = df.iloc[start_idx:end_idx].copy()
            shards.append({
                'shard_id': i,
                'data': shard_df[['sequence', 'hash_index']].to_dict('records'),
                'start_idx': start_idx,
                'end_idx': end_idx
            })
    
    print(f"Data divided into {len(shards)} shards")
    
    # Use multiprocessing to process each shard in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all shard tasks
        future_to_shard = {}
        for shard in shards:
            future = executor.submit(
                process_data_shard,
                shard['shard_id'],
                shard['data'],
                args,
                dtype
            )
            future_to_shard[future] = shard['shard_id']
        
        # Collect results
        all_predictions = []
        for future in as_completed(future_to_shard):
            shard_id = future_to_shard[future]
            try:
                shard_predictions = future.result()
                all_predictions.extend(shard_predictions)
                print(f"Shard {shard_id} completed, collected {len(shard_predictions)} predictions")
            except Exception as e:
                print(f"Shard {shard_id} generated an exception: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"All shards completed in {elapsed_time:.2f} seconds")
    
    # Merge results
    print("Merging results...")
    
    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(all_predictions)
    
    # Merge predictions back to original DataFrame using hash_index as key
    results_df = df.merge(pred_df, on='hash_index', how='left', suffixes=('', '_pred'))
    
    # Check for missing predictions
    missing_count = results_df['pred'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} sequences missing predictions")
        results_df['pred'] = results_df['pred'].fillna("")
    
    # Verify merged data order
    if results_df['label'].equals(df['label']) and results_df['type'].equals(df['type']):
        print("✅ Merge verification passed: all data maintains original order")
    else:
        print("⚠️ Merge verification failed: data order may be corrupted")
    
    # Calculate accuracy
    final_predictions = results_df['pred'].tolist()
    final_labels = results_df['label'].tolist()
    
    accuracies = calculate_accuracy(final_predictions, final_labels)
    results_df['accuracy'] = accuracies
    
    # Calculate statistics
    type_means = results_df.groupby('type')['accuracy'].mean()
    overall_mean = results_df['accuracy'].mean()

    # Save detailed results to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model_path.split('/')[-1]
    output_filename = f"{model_name}_{dtype}.parquet"
    output_path = os.path.join(args.output_dir, output_filename)
    results_df[['pred', 'label', 'type', 'accuracy']].to_parquet(output_path)
    
    print(f"Checkpoint {model_name} ({dtype}) - Overall Accuracy: \n{overall_mean:.4f}")
    print(f"Checkpoint {model_name} ({dtype}) - Type-wise Accuracy: \n{type_means}")
    print("-" * 80)
    print(f"✅ Completed {model_name} with {dtype}")
    print(f"📊 Results saved to: {output_path}")
    print("-" * 80)

def main():
    args = parse_args()
    dtype = "bfloat16" if args.bf16 else "float32"
    process_checkpoint(args, dtype)

if __name__ == "__main__":
    main()
