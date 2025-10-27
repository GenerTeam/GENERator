import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import multiprocessing as mp
import time
from typing import Dict, List, Tuple
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for variant effect prediction.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Downstream Task: Variant Effect Prediction"
    )
    parser.add_argument(
        "--hg38_path",
        type=str,
        default="hf://datasets/GenerTeam/variant-effect-prediction/test.parquet",
        help="Path to hg38 reference genome parquet file",
    )
    parser.add_argument(
        "--clinvar_path",
        type=str,
        default="hf://datasets/GenerTeam/variant-effect-prediction/ClinVar_VEP_results.parquet",
        help="Path to ClinVar variants parquet file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="GenerTeam/GENERator-eukaryote-1.2b-base",
        help="HuggingFace model path or name",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size for model inference"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=32,
        help="Number of processes for parallel computation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/variant_effect_predictions.parquet",
        help="Path to save the output predictions",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=96000,
        help="Context length in base pairs (bp) for sequence extraction",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 for faster inference, otherwise use fp32",
    )
    return parser.parse_args()

def load_and_prepare_data(
    hg38_path: str, clinvar_path: str, context_length: int
) -> pd.DataFrame:
    """
    Load genomic data and prepare sequences for variant effect prediction.

    Args:
        hg38_path: Path to the hg38 reference genome parquet file
        clinvar_path: Path to the ClinVar variants parquet file
        context_length: Context length in base pairs (bp) for sequence extraction

    Returns:
        DataFrame with variants and their context sequences
    """
    print("🧬 Loading genomic data...")
    start_time = time.time()
    seq_df = pd.read_parquet(hg38_path)
    clinvar_df = pd.read_parquet(clinvar_path)

    print(f"📊 Loaded {len(clinvar_df)} ClinVar variants")
    print(f"⚡ Data loading completed in {time.time() - start_time:.2f} seconds")

    print("🧪 Extracting sequences for each variant...")
    sequence_start_time = time.time()
    sequences = []
    for i in tqdm(range(len(clinvar_df)), desc="Sequence Extraction"):
        chrom_id = clinvar_df["chrom"][i]
        location = clinvar_df["pos"][i] - 1

        # Extract sequence - context_length bp upstream of the variant position
        sequence = seq_df.loc[seq_df["ID"] == "chr" + chrom_id]["Sequence"].values[0][
            max(0, location - context_length) : location
        ]

        # Remove leading 'N' characters if present
        sequence = sequence.lstrip("N")

        # Ensure sequence length is divisible by 6 for 6-mer tokenizer
        truncate_length = len(sequence) % 6
        if truncate_length > 0:
            sequence = sequence[truncate_length:]

        sequences.append(sequence)

    clinvar_df["sequence"] = sequences
    
    # Generate unique hash indices for each sequence
    print("Generating hash indices for sequences...")
    clinvar_df['hash_index'] = clinvar_df.apply(
        lambda row: hashlib.md5(f"{row['sequence']}_{row.name}".encode()).hexdigest()[:16], 
        axis=1
    )
    
    print(
        f"✅ Sequence extraction completed in {time.time() - sequence_start_time:.2f} seconds"
    )
    print(f"📏 Average sequence length: {np.mean([len(s) for s in sequences]):.1f} bp")

    return clinvar_df

def compute_logits_shard(args):
    """Compute logits shard on a single GPU"""
    shard_id, sequences_data, model_path, dtype, batch_size = args
    
    # Set current GPU
    torch.cuda.set_device(shard_id)
    device = f"cuda:{shard_id}"
    
    print(f"Shard {shard_id}: Loading model on GPU {shard_id}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        dtype=getattr(torch, dtype)
    ).to(device)
    
    model.eval()
    
    # Extract sequence data
    sequences_shard = [item['sequence'] for item in sequences_data]
    indices_shard = [item['hash_index'] for item in sequences_data]
    total_sequences = len(sequences_shard)
    
    logits_shard = []
    
    with tqdm(total=total_sequences, desc=f"Shard {shard_id}", unit="seq") as pbar:
        for i in range(0, total_sequences, batch_size):
            batch_sequences = sequences_shard[i:i + batch_size]
            batch_indices = indices_shard[i:i + batch_size]

            # Tokenize sequences
            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)

            # Get logits for the last token in each sequence
            for j, seq in enumerate(batch_sequences):
                seq_len = len(tokenizer(seq).input_ids)
                last_token_logits = outputs.logits[j, seq_len - 2, :]
                
                # Apply softmax to get probabilities
                probs = F.softmax(last_token_logits, dim=0).cpu().float().numpy().tolist()
                logits_shard.append({
                    "hash_index": batch_indices[j],
                    "logits": probs
                })

            pbar.update(len(batch_sequences))
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return logits_shard

def compute_logits_parallel(
    clinvar_df: pd.DataFrame,
    model_path: str,
    dtype: str,
    batch_size: int = 32
) -> List[List[float]]:
    """
    Compute logits using multi-GPU parallel processing
    
    Args:
        clinvar_df: DataFrame with variant information
        model_path: Path to the model
        dtype: Data type (bfloat16 or float32)
        batch_size: Batch size per GPU

    Returns:
        List of softmax probabilities for next token prediction
    """
    print("🧠 Computing logits using parallel GPU processing...")
    start_time = time.time()
    
    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for parallel computation")
    
    # Prepare data
    sequences_data = clinvar_df[['sequence', 'hash_index']].to_dict('records')
    total_sequences = len(sequences_data)
    
    # Split data into multiple shards
    shard_size = (total_sequences + num_gpus - 1) // num_gpus
    shards = []
    
    for i in range(num_gpus):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_sequences)
        if start_idx < total_sequences:
            shards.append({
                'shard_id': i,
                'sequences_data': sequences_data[start_idx:end_idx],
                'start_idx': start_idx,
                'end_idx': end_idx
            })
    
    print(f"Data divided into {len(shards)} shards")
    
    # Prepare arguments
    args_list = []
    for shard in shards:
        args_list.append((
            shard['shard_id'],
            shard['sequences_data'],
            model_path,
            dtype,
            batch_size
        ))
    
    # Use multiprocessing to process each shard in parallel
    all_logits_dict = {}
    
    # Use spawn context to avoid CUDA multiprocessing issues
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        results = list(tqdm(
            pool.imap(compute_logits_shard, args_list),
            total=len(args_list),
            desc="Processing Shards"
        ))
    
    # Merge results into dictionary
    for shard_result in results:
        for item in shard_result:
            all_logits_dict[item['hash_index']] = item['logits']
    
    # Reconstruct logits list in original DataFrame order
    all_logits = [all_logits_dict[hash_index] for hash_index in clinvar_df['hash_index']]
    
    # Verify all logits have been collected
    missing_count = len([x for x in all_logits if x is None])
    if missing_count > 0:
        print(f"Warning: {missing_count} sequences missing logits")
        # Get vocabulary size and fill missing values
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        vocab_size = len(tokenizer)
        for i in range(len(all_logits)):
            if all_logits[i] is None:
                all_logits[i] = [0.0] * vocab_size
    
    print(f"✅ Parallel logit computation completed in {time.time() - start_time:.2f} seconds")
    return all_logits

def get_char_indices(vocab: Dict[str, int]) -> Dict[str, List[int]]:
    """
    Create a mapping from characters to their token indices.

    Args:
        vocab: The tokenizer vocabulary

    Returns:
        Dictionary mapping first characters to their token indices
    """
    tokens = list(vocab.keys())
    token_ids = list(vocab.values())

    sorted_pairs = sorted(zip(token_ids, tokens))
    sorted_tokens = [token for _, token in sorted_pairs]

    char_indices = {}
    for i, token in enumerate(sorted_tokens):
        if isinstance(token, str) and len(token) > 0:
            first_char = token[0]
            if first_char not in char_indices:
                char_indices[first_char] = []
            char_indices[first_char].append(i)

    return char_indices

def compute_prob(
    args: Tuple[str, str, List[float], Dict[str, List[int]]]
) -> Tuple[float, float]:
    """
    Compute probabilities for reference and alternate alleles.

    Args:
        args: Tuple containing (ref, alt, logits, char_indices)

    Returns:
        Tuple of (reference probability, alternate probability)
    """
    ref, alt, logits, char_indices = args
    p_ref = sum(logits[i] for i in char_indices.get(ref, []) if i < len(logits))
    p_alt = sum(logits[i] for i in char_indices.get(alt, []) if i < len(logits))
    return p_ref, p_alt

def parallel_compute_probabilities(
    clinvar_df: pd.DataFrame,
    logits: List[List[float]],
    tokenizer: PreTrainedTokenizer,
    num_processes: int = 16,
) -> Tuple[List[float], List[float]]:
    """
    Compute reference and alternate probabilities.

    Args:
        clinvar_df: DataFrame with variant information
        logits: List of logits for each variant
        tokenizer: Tokenizer with vocabulary
        num_processes: Number of parallel processes

    Returns:
        Lists of reference and alternate probabilities
    """
    print(f"🧮 Computing variant probabilities with {num_processes} processes...")
    start_time = time.time()

    # Get vocabulary directly from tokenizer
    vocab = tokenizer.get_vocab()
    char_indices = get_char_indices(vocab)

    # Prepare arguments for parallel processing
    args_list = [
        (clinvar_df["ref"][i], clinvar_df["alt"][i], logits[i], char_indices)
        for i in range(len(clinvar_df))
    ]

    # Run parallel computation with larger chunksize for better efficiency
    chunksize = max(1, len(args_list) // (num_processes * 4))
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(compute_prob, args_list, chunksize=chunksize),
                total=len(args_list),
                desc="Computing Probabilities",
            )
        )

    # Unpack results
    p_ref, p_alt = zip(*results)
    print(
        f"✅ Probability computation completed in {time.time() - start_time:.2f} seconds"
    )
    return list(p_ref), list(p_alt)

def evaluate_predictions(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Evaluate variant effect predictions using AUROC and AUPRC.

    Args:
        labels: True variant labels (pathogenic/benign)
        scores: Predicted variant scores

    Returns:
        Dictionary with evaluation metrics
    """
    print("📊 Evaluating model predictions...")
    start_time = time.time()

    # Calculate AUROC
    auroc = roc_auc_score(labels, scores)

    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision)

    print(f"⏱️ Evaluation completed in {time.time() - start_time:.2f} seconds")
    return {"AUROC": auroc, "AUPRC": auprc}

def save_results(df: pd.DataFrame, path: str) -> None:
    """
    Save results to a parquet file.

    Args:
        df: DataFrame with results
        path: Path to save the output file
    """
    print(f"💾 Saving predictions to {path}")
    start_time = time.time()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)

    print(f"✅ Results saved in {time.time() - start_time:.2f} seconds")
    print(f"📊 Saved {len(df)} variant predictions")

def display_progress_header() -> None:
    """
    Display a stylized header for the variant effect prediction.
    """
    print("\n" + "=" * 80)
    print("🧬  VARIANT EFFECT PREDICTION PIPELINE  🧬")
    print("=" * 80 + "\n")

def main() -> None:
    """
    Main function to run the variant effect prediction pipeline.
    """
    # Display header
    display_progress_header()

    # Start timer for total execution
    total_start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Load and prepare data with user-specified context length
    clinvar_df = load_and_prepare_data(
        args.hg38_path, args.clinvar_path, args.context_length
    )

    dtype = "bfloat16" if args.bf16 else "float32"

    # Load tokenizer (for subsequent probability calculation)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Compute logits using parallel processing
    logits = compute_logits_parallel(
        clinvar_df,
        args.model_path,
        dtype,
        batch_size=args.batch_size
    )

    # Compute probabilities for reference and alternate alleles
    p_ref, p_alt = parallel_compute_probabilities(
        clinvar_df, logits, tokenizer, num_processes=args.num_processes
    )

    # Add results to DataFrame
    clinvar_df["p_ref"] = p_ref
    clinvar_df["p_alt"] = p_alt

    # Calculate scores and prepare for evaluation
    clinvar_df["label"] = clinvar_df["label"].astype(int)
    clinvar_df["score"] = np.log(clinvar_df["p_ref"] / (clinvar_df["p_alt"] + 1e-10))

    # Evaluate predictions
    metrics = evaluate_predictions(
        clinvar_df["label"].values, clinvar_df["score"].values
    )

    # Print results
    print("\n" + "=" * 80)
    print(f"🏆 EVALUATION RESULTS FOR {args.model_path.split('/')[-1]} ({dtype}) 🏆")
    print("=" * 80)
    print(f"🎯 AUROC: {metrics['AUROC']:.4f}")
    print(f"📈 AUPRC: {metrics['AUPRC']:.4f}")
    print("=" * 80)

    save_results(clinvar_df.drop(columns=["sequence", "hash_index"]), args.output_path)

    # Print total execution time
    total_time = time.time() - total_start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n⏱️ Total execution time: {int(minutes)}m {seconds:.2f}s")
    print("✨ Completed successfully! ✨\n")

if __name__ == "__main__":
    main()
