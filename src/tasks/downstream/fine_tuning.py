import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_DISABLE_XET"] = "1"
# os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"

import argparse
import wandb
import logging
import torch

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, DatasetDict, load_dataset

import sys
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "src"))

from custom_trainer import BPTrainer
from custom_dataset import ParquetSequenceDataset, SequenceDataCollator

config_dir = ROOT_DIR / "configs"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GENERator model")
    # Data & model
    parser.add_argument("--model_name", type=str, default="GenerTeam/GENERator-v2-eukaryote-1.2b-base",
        help="HuggingFace model path or name",
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--parquet_path", type=str, default=None,
        help="Parquet file or directory (must include sequence column)",
    )
    data_group.add_argument("--dataset_name", type=str, default=None,
        help="HuggingFace dataset name/path (loaded via datasets.load_dataset)",
    )
    parser.add_argument("--subset_name", type=str, default=None,
        help="Dataset subset/config name for --dataset_name (optional)",
    )
    parser.add_argument("--dataset_split", type=str, default="train",
        help="Dataset split to use for training",
    )
    parser.add_argument("--sequence_col", type=str, default="sequence",
        help="Sequence column name",
    )

    # Model runtime
    parser.add_argument("--max_token_length", type=int, default=16384,
        help="Maximum sequence length in tokens",
    )
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", 
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation: flash_attention_2, sdpa, eager",
    )
    parser.add_argument("--bf16", action='store_true',
        help="Enable BF16 (recommended for A100)",
    )

    # Run / outputs
    parser.add_argument("--output_dir", type=str, default=None,
        help="Training checkpoints output directory",
    )
    parser.add_argument("--saved_model_dir", type=str, default=None,
        help="Final model output directory",
    )
    parser.add_argument("--tmp_dir", type=str, default=None,
        help="Temporary directory (sets TMPDIR)",
    )

    # Training schedule
    parser.add_argument("--epochs", type=float, default=3,
        help="Training epochs",
    )
    parser.add_argument("--batch_size", type=int, default=4,
        help="Per-device batch size",
    )
    parser.add_argument("--gradient_accumulation", type=int, default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--lr", type=float, default=5e-5,
        help="Learning rate",
    )
    parser.add_argument("--warm_up", type=int, default=2000,
        help="Warmup steps",
    )
    parser.add_argument("--save_steps", type=int, default=2000,
        help="Steps between checkpoints",
    )
    parser.add_argument("--save_total_limit", type=int, default=10,
        help="Max number of checkpoints",
    )
    parser.add_argument("--logging_steps", type=int, default=10,
        help="Logging steps",
    )

    # Logging / tracking
    parser.add_argument("--report_to", type=str, default="wandb",
        help="Reporting tool",
    )
    parser.add_argument("--wandb_project", type=str, default="GENERator-finetuning",
        help="W&B project name",
    )
    parser.add_argument("--wandb_key", type=str, default=None,
        help="W&B API key (optional)",
    )
    parser.add_argument("--run_name", type=str, default=None,
        help="W&B run name",
    )

    # Distributed training
    parser.add_argument("--distributed_type", type=str, default="ddp", choices=["ddp", "fsdp", "deepspeed"],
        help="Type of distributed training to use",
    )
    parser.add_argument("--fsdp_config", type=str, default=str(config_dir / "distributed_configs" / "fsdp_config.json"),
        help="FSDP config path",
    )
    parser.add_argument("--ds_config", type=str, default=str(config_dir / "distributed_configs" / "ds_config.json"),
        help="DeepSpeed config path",
    )
    parser.add_argument("--local_rank", type=int, default=0,
        help="Local rank for distributed training",
    )
    
    return parser.parse_args()

def resolve_precision(args):
    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()

    if args.bf16 and not bf16_supported:
        if not cuda_available:
            logger.warning("BF16 requested but CUDA is not available; using FP32.")
        else:
            logger.warning("BF16 requested but not supported on this hardware; using FP32.")
        args.bf16 = False

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    logger.info("Using %s precision.", "BF16" if args.bf16 else "FP32")
    return dtype

def should_use_wandb(report_to):
    if isinstance(report_to, (list, tuple)):
        targets = {str(x).lower() for x in report_to}
    else:
        targets = {s.strip().lower() for s in str(report_to).split(",")}
    return "wandb" in targets

def setup_logging_and_wandb(args):
    if not should_use_wandb(args.report_to):
        logger.info(f"W&B disabled (report_to={args.report_to}).")
        return

    if str(os.environ.get("WANDB_DISABLED", "")).lower() in {"1", "true", "yes"}:
        logger.info("W&B disabled via WANDB_DISABLED.")
        return

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.run_name

    env_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
    has_key = bool(args.wandb_key or env_key)

    if has_key:
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = "online"
        if args.wandb_key:
            wandb.login(key=args.wandb_key, force=True)
        else:
            wandb.login(force=True)
        logger.info(f"W&B online, project: {args.wandb_project}, run: {args.run_name}")
    else:
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = "offline"
        logger.info("W&B offline mode (no API key provided).")

def get_training_args(args):
    kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        fp16=False,
        bf16=args.bf16,
        max_grad_norm=1.0,
        
        # Optimizer & learning rate
        run_name=args.run_name,
        warmup_steps=args.warm_up,
        weight_decay=0.01,
        learning_rate=args.lr,
        lr_scheduler_type='cosine_with_min_lr',
        lr_scheduler_kwargs={'min_lr_rate': 0.1},
        
        # Gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False},
        
        # Dataset columns
        remove_unused_columns=False,
    )

    # Distributed training
    if args.distributed_type == "fsdp":
        kwargs['fsdp'] = "shard_grad_op auto_wrap"
        kwargs['fsdp_config'] = args.fsdp_config
    elif args.distributed_type == "deepspeed":
        kwargs['deepspeed'] = args.ds_config
    
    return TrainingArguments(**kwargs)

def load_model(args, dtype):
    """Try attention backends in list order starting from the requested one."""
    order = ["flash_attention_2", "sdpa", "eager"]
    requested = args.attn_implementation
    if requested not in order:
        raise ValueError(f"Unsupported attention implementation: {args.attn_implementation}")

    tried = []
    for impl in order[order.index(requested):]:
        try:
            if impl == "flash_attention_2" and dtype == torch.float32:
                raise ValueError("flash_attention_2 requires FP16/BF16 (got FP32)")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                attn_implementation=impl,
                dtype=dtype,
                trust_remote_code=True,
            )
            if impl != requested:
                logger.warning(f"Falling back to attention implementation: {impl}")
            return model
        except Exception as e:
            tried.append((impl, str(e)))
            logger.warning(f"Failed to load model with attn_implementation={impl}: {e}")

    details = "; ".join([f"{impl}: {err}" for impl, err in tried])
    raise RuntimeError(f"Failed to load model with any attention backend. {details}")

def load_hf_sequence_dataset(
    dataset_name: str,
    subset_name: Optional[str],
    dataset_split: Optional[str],
    sequence_col: str,
):
    if subset_name is None:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, subset_name, trust_remote_code=True)

    split: Optional[str] = None
    if isinstance(dataset, DatasetDict):
        split = dataset_split or "train"
        if split not in dataset:
            raise ValueError(
                f"Unable to pick a training split for dataset '{dataset_name}'. "
                f"Requested split: {split}. Available splits: {list(dataset.keys())}. "
                "Please set --dataset_split."
            )
        dataset = dataset[split]

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Unexpected dataset type returned by load_dataset: {type(dataset)}")

    if sequence_col not in set(dataset.column_names):
        raise ValueError(
            f"sequence_col '{sequence_col}' not found in dataset columns: {dataset.column_names}"
        )
    logger.info(
        "Loaded HF dataset: name=%s subset=%s split=%s sequence_col=%s",
        dataset_name,
        subset_name,
        split,
        sequence_col,
    )

    def _process(batch: Dict[str, Any]) -> Dict[str, Any]:
        seqs = batch[sequence_col]
        processed = []
        for s in seqs:
            if s is None:
                processed.append("")
                continue
            processed.append(str(s).strip().upper())
        return {"text": processed}

    return dataset.map(_process, batched=True, remove_columns=dataset.column_names)

def load_train_dataset(args):
    if args.parquet_path is not None:
        return ParquetSequenceDataset(
            parquet_path=args.parquet_path,
            sequence_col=args.sequence_col,
            uppercase=True,
            strip=True,
        )

    if args.dataset_name is not None:
        return load_hf_sequence_dataset(
            dataset_name=args.dataset_name,
            subset_name=args.subset_name,
            dataset_split=args.dataset_split,
            sequence_col=args.sequence_col,
        )

    raise ValueError("Either --parquet_path or --dataset_name must be provided.")

def main():
    args = parse_args()

    if args.tmp_dir is not None:
        os.environ["TMPDIR"] = args.tmp_dir

    # Resolve run name
    if args.run_name is None:
        args.run_name = f"GENERator-finetuning_{args.max_token_length}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # Resolve output directories
    if args.output_dir is None:
        args.output_dir = str(Path("checkpoints") / args.run_name)
    if args.saved_model_dir is None:
        args.saved_model_dir = str(Path("saved_model") / args.run_name)

    if args.distributed_type == "fsdp" and not Path(args.fsdp_config).exists():
        raise FileNotFoundError(f"FSDP config not found: {args.fsdp_config}")
    if args.distributed_type == "deepspeed" and not Path(args.ds_config).exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {args.ds_config}")
    
    # W&B setup
    setup_logging_and_wandb(args)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Create model with requested attention backend and precision (with fallbacks)
    model_dtype = resolve_precision(args)
    model = load_model(args, model_dtype)
    model.vocab_size = tokenizer.vocab_size

    # Prepare data
    logger.info("Preparing data...")
    train_dataset = load_train_dataset(args)
    logger.info(f"Dataset size: {len(train_dataset)}")

    data_collator = SequenceDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_token_length,
        pad_to_multiple_of=None,
        add_special_tokens=True,  # Let tokenizer add <s> and </s>
    )
    
    # Training arguments
    training_args = get_training_args(args)
    
    # Auto-resume
    output_dir_path = Path(args.output_dir)
    resume_from_checkpoint = (
        get_last_checkpoint(str(output_dir_path)) if output_dir_path.exists() else None
    )
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    
    # Initialize Trainer
    trainer = BPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        bp_loss_only=True,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save model
    logger.info(f"Saving final model to {args.saved_model_dir}")
    acc = trainer.accelerator
    acc.wait_for_everyone()

    if acc.distributed_type.name == "FSDP":
        acc.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    unwrapped = acc.unwrap_model(trainer.model)

    unwrapped.save_pretrained(
        args.saved_model_dir,
        is_main_process=acc.is_main_process,
        save_function=acc.save,
        state_dict=acc.get_state_dict(trainer.model),
        safe_serialization=True,
    )

    if acc.is_main_process:
        tokenizer.save_pretrained(args.saved_model_dir)

    acc.wait_for_everyone()

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
