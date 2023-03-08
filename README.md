# LLaMA_MPS
Run LLaMA inference on Apple Silicon GPUs.

![Demo](demo.gif)

### Setup

**1. Clone this repo**

`git clone https://github.com/jankais3r/LLaMA_MPS`

**2. [Download the model weights](https://github.com/facebookresearch/llama/pull/73/files#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5R4) and put them into a folder called** `models` (e.g., `LLaMA_MPS/models/7B`)

**3. Install Python dependencies**

```bash
pip3 install -r requirements.txt
pip3 install -e .
```

**4. _(Optional)_ Reshard the model (13B/30B/65B)**

Since we are running the inference on a single GPU, we need to merge the larger models' weights into a single file.

```bash
mv models/13B models/13B_orig
mkdir models/13B
python3 reshard.py 1 models/13B_orig models/13B
```

**5. Run the inference**

`PYTORCH_ENABLE_MPS_FALLBACK=1 torchrun chat.py --ckpt_dir models/13B  --tokenizer_path models/tokenizer.model --max_batch_size=8`

### Memory requirements

| Model  | Peak memory during load | Memory during inference |
| ------------- | ------------- | ------------- |
| 7B  | 25 GB  | 16 GB  |
| 13B  | 66 GB  | 32 GB  |
| 30B  | ?? GB  | ?? GB  |
| 65B  | ?? GB  | ?? GB  |

### Parameters to experiment with
**- max_batch_size**

If you are running low on memory, you can reduce `max_batch_size` to 1 (by simply ommiting the `--max_batch_size=8` argument).

**- max_gen_len**

To increase/reduce the length of the generated text, edit the `max_gen_len` value in [chat.py](https://github.com/jankais3r/LLaMA_MPS/blob/main/chat.py#L83).

**- use_repetition_penalty**

The example script now has implemented penalty for generating repeated content. This should lead to higher quality output, but it slightly slows down the inference. Run the script with `--use_repetition_penalty=False` argument to disable the penalty algorithm.


### Credits

- facebookresearch ([original code](https://github.com/facebookresearch/llama))
- markasoftware ([cpu optimizations](https://github.com/markasoftware/llama-cpu))
- remixer-dec ([mps optimizations](https://github.com/remixer-dec/llama-mps))
- venuatu ([continuous token printing](https://github.com/venuatu/llama/commit/25c84973f71877677547453dab77eeaea9a86376))
- benob ([reshard script](https://gist.github.com/benob/4850a0210b01672175942203aa36d300))
- tloen ([repetition penalty](https://github.com/tloen/llama-int8))
