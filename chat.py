#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import warnings

warnings.filterwarnings("ignore")
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import resource

resource.setrlimit(resource.RLIMIT_NOFILE, (10000, 10000))

import sys
import fire
import time
import json
import torch
import random
import pyarrow as pa
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    arrow_dir = Path(ckpt_dir).expanduser() / "arrow"
    if not arrow_dir.exists():
        checkpoints = sorted(Path(ckpt_dir).expanduser().glob("*.pth"))
        if len(checkpoints) > 1:
            print(
                "The selected model is split into several checkpoints and needs to be merged first.\nUse the 'reshard.py' script."
            )
            sys.exit()
        print("Converting checkpoint to pyarrow format")
        for ckpt_file in checkpoints:
            print(ckpt_file)
            index = ckpt_file.parts[-1].split(".")[-2]

            ckpt = torch.load(ckpt_file, map_location="cpu")
            (arrow_dir / index).mkdir(parents=True, exist_ok=True)
            for k, v in ckpt.items():
                tens = pa.Tensor.from_numpy(v.numpy())
                with pa.output_stream(arrow_dir / index / k) as f:
                    pa.ipc.write_tensor(tens, f)
            ckpt = None
        print(
            "Checkpoint converted - feel free to delete the original '.pth' file (while keeping the 'arrow' folder)"
        )

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    random_seed = random.randint(1, 65534)
    torch.manual_seed(random_seed)
    print(f"Seed: {random_seed:5d}")
    start_time = time.time()
    print("Loading checkpoint")
    segments = sorted((arrow_dir / "00").glob("*"))

    checkpoint = {}
    files = []
    for seg in segments:
        f = pa.memory_map(str(seg))
        files.append(f)
        t = pa.ipc.read_tensor(f).to_numpy()
        t = torch.from_numpy(t)
        checkpoint[seg.parts[-1]] = t
        f.close()

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to("mps")

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.75,
    use_repetition_penalty: bool = True,
    repetition_penalty_range: int = 1024,
    repetition_penalty_slope: float = 0,
    repetition_penalty: float = 1.15,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
):

    generator = load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    
    if 'B-alpaca' in ckpt_dir:
        alpaca_mode = True
        print("Running the fine-tuned 'alpaca' model in an instruction-response mode.")
    else:
        alpaca_mode = False
        print("Running the raw 'llama' model in an auto-complete mode.")

    try:
        while True:
            if alpaca_mode:
                queryInputs = [
                    f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{input("Instruction: ")}
### Response:"""
                ]
                print("Response: ", end="")
            else:
                queryInputs = [input("Enter your LLaMA prompt: ")]
                print("Thinking...")
            queryTime = time.time()
            results = generator.generate(
                queryInputs,
                max_gen_len=max_seq_len,
                temperature=temperature,
                top_p=top_p,
                use_repetition_penalty=use_repetition_penalty,
                repetition_penalty_range=repetition_penalty_range,
                repetition_penalty_slope=repetition_penalty_slope,
                repetition_penalty=repetition_penalty,
            )
            print(f"\n\nInferred in {time.time() - queryTime:.2f} seconds")
            print("==================================\n")
    except KeyboardInterrupt:
        sys.exit()


if __name__ == "__main__":
    fire.Fire(main)
