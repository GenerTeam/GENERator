import argparse
import multiprocessing as mp
import os
import time
from typing import Dict, List, Tuple, Union

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
    PreTrainedModel,
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
        default="hf://datasets/songlab/clinvar/test.parquet",
        help="Path to ClinVar variants parquet file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GenerTeam/GENERator-eukaryote-1.2b-base",
        help="HuggingFace model path or name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for model inference"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=32,
        help="Number of processes for parallel computation",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Number of GPUs to use for DataParallel (1 for single GPU)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/variant_predictions.parquet",
        help="Path to save the output predictions",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=96000,
        help="Context length in base pairs (bp) for sequence extraction",
    )
    return parser.parse_args()


def extract_sequence(args: Tuple[str, int, int, pd.DataFrame]) -> str:
    """
    Extract sequence for a single variant.

    Args:
        args: Tuple containing (chrom_id, position, context_length, seq_df)

    Returns:
        Processed sequence
    """
    chrom_id, position, context_length, seq_df = args
    location = position - 1

    # Extract sequence upstream of the variant position
    sequence = seq_df.loc[seq_df["ID"] == "chr" + chrom_id]["Sequence"].values[0][
        max(0, location - context_length) : location
    ]

    # Remove leading 'N' characters if present
    sequence = sequence.lstrip("N")

    # Ensure sequence length is divisible by 6 for 6-mer tokenizer
    truncate_length = len(sequence) % 6
    if truncate_length > 0:
        sequence = sequence[truncate_length:]

    return sequence


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
    print(
        f"✅ Sequence extraction completed in {time.time() - sequence_start_time:.2f} seconds"
    )
    print(f"📏 Average sequence length: {np.mean([len(s) for s in sequences]):.1f} bp")

    return clinvar_df


def setup_model(
    model_name: str, dp_size: int = 1
) -> Tuple[
    Union[PreTrainedModel, torch.nn.DataParallel], PreTrainedTokenizer, torch.device
]:
    """
    Load and setup the model with optional multi-GPU support.

    Args:
        model_name: Name or path of the HuggingFace model
        dp_size: Number of GPUs to use for DataParallel (1 for single GPU)

    Returns:
        tuple of (model, tokenizer, device)
    """
    print(f"🤗 Loading model from Hugging Face: {model_name}")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float32"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )

    # Check available GPUs
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if dp_size > 1 and available_gpus >= dp_size:
            print(f"🚀 Using DataParallel with {dp_size} GPUs")
            model = torch.nn.DataParallel(model, device_ids=list(range(dp_size)))
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:0")
            if dp_size > 1:
                print(f"⚠️ Requested {dp_size} GPUs but only {available_gpus} available")
                print(f"🔄 Using single GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"💻 Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU available, using CPU")

    model.to(device)
    print(f"⏱️ Model loading completed in {time.time() - start_time:.2f} seconds")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model size: {total_params/1e6:.1f}M parameters")

    return model, tokenizer, device


def compute_logits(
    model: Union[PreTrainedModel, torch.nn.DataParallel],
    tokenizer: PreTrainedTokenizer,
    sequences: List[str],
    device: torch.device,
    batch_size: int = 4,
) -> List[List[float]]:
    """
    Compute logits for each sequence using the specified model.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        sequences: List of DNA sequences
        device: Computation device
        batch_size: Batch size for inference

    Returns:
        List of softmax probabilities for next token prediction
    """
    print("🧠 Computing logits using pre-trained model...")
    model.eval()
    start_time = time.time()

    all_logits: List[List[float]] = []

    # Adjust batch size based on available GPUs if using DataParallel
    if isinstance(model, torch.nn.DataParallel):
        effective_batch_size = batch_size * len(model.device_ids)
        print(
            f"⚡ Adjusted batch size: {effective_batch_size} (base: {batch_size} × {len(model.device_ids)} GPUs)"
        )
        batch_size = effective_batch_size

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing Batches"):
        batch_sequences = sequences[i : i + batch_size]

        # Tokenize sequences
        inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits for the last token in each sequence
        for j, seq in enumerate(batch_sequences):
            seq_len = len(tokenizer(seq).input_ids)

            # Handle DataParallel output if needed
            if isinstance(outputs.logits, torch.nn.parallel.scatter_gather.Scatter):
                last_token_logits = outputs.logits.to_local[0][j, seq_len - 2, :]
            else:
                last_token_logits = outputs.logits[
                    j, seq_len - 2, :
                ]  # -2 because of the EOS token

            # Apply softmax to get probabilities
            probs = F.softmax(last_token_logits, dim=0).cpu().float().numpy().tolist()
            all_logits.append(probs)

    print(f"✅ Logit computation completed in {time.time() - start_time:.2f} seconds")
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

    # Setup model and tokenizer with specified DP size
    model, tokenizer, device = setup_model(args.model_name, args.dp_size)

    # Compute logits for each sequence
    logits = compute_logits(
        model,
        tokenizer,
        clinvar_df["sequence"].tolist(),
        device=device,
        batch_size=args.batch_size,
    )

    # Free up GPU memory
    if torch.cuda.is_available():
        print("🧹 Cleaning up GPU memory...")
        torch.cuda.empty_cache()
        if isinstance(model, torch.nn.DataParallel):
            model.module = model.module.cpu()
        else:
            model.cpu()
        print("✅ GPU memory cleaned")

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
    print(f"🏆 EVALUATION RESULTS FOR {args.model_name} 🏆")
    print("=" * 80)
    print(f"🎯 AUROC: {metrics['AUROC']:.4f}")
    print(f"📈 AUPRC: {metrics['AUPRC']:.4f}")
    print("=" * 80)

    # Save results to parquet file
    save_results(clinvar_df, args.output_path)

    # Print total execution time
    total_time = time.time() - total_start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n⏱️ Total execution time: {int(minutes)}m {seconds:.2f}s")
    print("✨ Completed successfully! ✨\n")


if __name__ == "__main__":
    main()
