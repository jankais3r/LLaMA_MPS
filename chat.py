# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import sys
import torch
import fire
import time
import json
import random
import warnings
from typing import Tuple
from pathlib import Path
warnings.filterwarnings("ignore")
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from fairscale.nn.model_parallel.initialize import initialize_model_parallel


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    random_seed = random.randint(1, 65534)
    torch.manual_seed(random_seed)
    print(f"Seed: {random_seed:5d}")
    return local_rank, world_size


def load(
    ckpt_dir: str, 
    tokenizer_path: str, 
    local_rank: int, 
    world_size: int, 
    max_seq_len: int, 
    max_batch_size: int, 
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP = {len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location = "cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len = max_seq_len, max_batch_size = max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path = tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict = False)
    model = model.to("mps")
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str, 
    tokenizer_path: str, 
    temperature: float = 0.8, 
    top_p: float = 0.95, 
    use_repetition_penalty = True, 
    repetition_penalty_range: int = 1024, 
    repetition_penalty_slope: float = 0, 
    repetition_penalty: float = 1.15, 
    max_seq_len: int = 512, 
    max_batch_size: int = 1):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    try:
        while True:
            queryInputs = [input("Enter your LLaMA prompt: ")]
            print("Thinking...")
            queryTime = time.time()
            results = generator.generate(
                queryInputs, 
                max_gen_len = 512, 
                temperature=temperature, 
                top_p = top_p, 
                use_repetition_penalty = use_repetition_penalty, 
                repetition_penalty_range = repetition_penalty_range, 
                repetition_penalty_slope = repetition_penalty_slope, 
                repetition_penalty = repetition_penalty
            )
            print(f"\nInferred in {time.time() - queryTime:.2f} seconds")
            print("==================================\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
