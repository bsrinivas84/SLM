# Load GPT-2 124M model from downloaded OpenAI checkpoint
# WITHOUT TensorFlow - uses raw checkpoint parsing via numpy/struct

import json
from pathlib import Path
import safetensors
import torch

import PreviousChapters

def load_gpt2_params_from_tf_ckpt_no_tf(model_dir):
    """
    Parse TensorFlow checkpoint files directly without importing TensorFlow.
    Reads the .index file to find variable names/shapes/offsets,
    then reads raw bytes from the .data file.
    """
    model_dir = Path(model_dir)
    data_file = model_dir / "model.ckpt.data-00000-of-00001"
    index_file = model_dir / "model.ckpt.index"

    # The .index file is a TensorBundle index (a string_table_builder SSTable).
    # We'll use a simpler approach: parse variable metadata from the index file.
    # TF checkpoint index is a protobuf-based format, but we can use the
    # bundled_tensor_slice approach.

    # Actually, the cleanest no-TF approach is to use the `ckpt` file format
    # parser. Let's parse the index manually.
    index_data = index_file.read_bytes()
    data_bytes = data_file.read_bytes()

    # Parse TF checkpoint index file (TensorBundle format)
    variables = _parse_tf_index(index_data, data_bytes)
    return variables


def _parse_tf_index(index_data, data_bytes):
    """
    Minimal parser for TF checkpoint .index files.
    Uses the fact that variable names are embedded as strings,
    and tensor data is stored contiguously in the .data file.
    """
    # The TF checkpoint index is an SSTable with protobuf entries.
    # For simplicity and reliability, we'll use a regex-like scan
    # to extract variable names and then use known GPT-2 structure
    # to determine shapes and read the data.

    # Better approach: use struct to parse the index table footer and entries.
    # However, this is complex. Instead, let's use numpy to load from
    # the raw data file using known GPT-2 124M architecture shapes.
    return _load_gpt2_124m_from_raw_data(data_bytes)


def _load_gpt2_124m_from_raw_data(data_bytes):
    """
    Load all float32 tensors from the checkpoint data file
    using the known GPT-2 124M architecture.
    This reads the entire data file as float32 and splits by known shapes.
    """
    # This is fragile. A better approach: use the `safetensors` or
    # Hugging Face `transformers` library to load GPT-2 weights.
    # Since we want NO external ML framework, let's use HuggingFace download
    # via simple HTTP + safetensors (numpy only).
    raise NotImplementedError("Raw .ckpt parsing is unreliable without TF protobuf.")


# GPT-2 model variants available on Hugging Face
GPT2_MODELS = {
    "gpt2-small": {"hf_name": "gpt2", "params": "124M"},
    "gpt2-medium": {"hf_name": "gpt2-medium", "params": "355M"},
    "gpt2-large": {"hf_name": "gpt2-large", "params": "774M"},
    "gpt2-xl": {"hf_name": "gpt2-xl", "params": "1558M"},
}


def download_gpt2_from_huggingface(model_dir, model_name="gpt2-small"):
    """
    Download GPT-2 weights from Hugging Face in safetensors format.
    
    Args:
        model_dir: Directory to save the downloaded files.
        model_name: One of 'gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
    """
    import requests

    if model_name not in GPT2_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(GPT2_MODELS.keys())}")

    hf_name = GPT2_MODELS[model_name]["hf_name"]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    hf_base = f"https://huggingface.co/openai-community/{hf_name}/resolve/main"
    files_to_download = {
        "config.json": f"{hf_base}/config.json",
        "model.safetensors": f"{hf_base}/model.safetensors",
    }

    for filename, url in files_to_download.items():
        dest = model_dir / filename
        if dest.exists():
            print(f"Already exists: {dest}")
            continue
        print(f"Downloading {filename}...")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  Saved: {dest}")

    return model_dir


def load_gpt2_safetensors(model_dir):
    """
    Load GPT-2 weights from safetensors file using PyTorch directly.
    Returns (settings, params_dict) where params_dict maps tensor names to torch tensors.
    """
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"

    settings = json.loads(config_path.read_text(encoding="utf-8"))
    params = load_file(str(weights_path))

    return settings, params


def settings_to_local_config(settings):
    """Map HuggingFace GPT-2 config keys to local GPTModel config keys."""
    return {
        "vocab_size": settings.get("vocab_size", 50257),
        "context_length": settings.get("n_positions", 1024),
        "emb_dim": settings.get("n_embd", 768),
        "n_heads": settings.get("n_head", 12),
        "n_layers": settings.get("n_layer", 12),
        "drop_rate": settings.get("resid_pdrop", 0.1),
        "qkv_bias": True,  # HF GPT-2 uses bias in attention projections
    }


def assign_gpt2_weights(model, params):
    """
    Assign HuggingFace GPT-2 safetensors weights to local GPTModel.
    Uses pure PyTorch tensors (no numpy needed).
    """
    import torch

    # Token and position embeddings
    model.tok_emb.weight.data = params["wte.weight"]
    model.pos_emb.weight.data = params["wpe.weight"]

    # Transformer blocks
    for i in range(len(model.trf_blocks)):
        block = model.trf_blocks[i]
        prefix = f"h.{i}."

        # Layer norms
        block.norm1.scale.data = params[f"{prefix}ln_1.weight"]
        block.norm1.shift.data = params[f"{prefix}ln_1.bias"]
        block.norm2.scale.data = params[f"{prefix}ln_2.weight"]
        block.norm2.shift.data = params[f"{prefix}ln_2.bias"]

        # Attention: HF stores c_attn as a single [emb_dim, 3*emb_dim] weight
        # and c_proj as [emb_dim, emb_dim]
        c_attn_w = params[f"{prefix}attn.c_attn.weight"]  # [emb_dim, 3*emb_dim]
        c_attn_b = params[f"{prefix}attn.c_attn.bias"]    # [3*emb_dim]

        # Split into Q, K, V
        q_w, k_w, v_w = c_attn_w.chunk(3, dim=1)
        q_b, k_b, v_b = c_attn_b.chunk(3, dim=0)

        block.att.W_query.weight.data = q_w.T.contiguous()
        block.att.W_query.bias.data = q_b
        block.att.W_key.weight.data = k_w.T.contiguous()
        block.att.W_key.bias.data = k_b
        block.att.W_value.weight.data = v_w.T.contiguous()
        block.att.W_value.bias.data = v_b

        # Output projection
        block.att.out_proj.weight.data = params[f"{prefix}attn.c_proj.weight"].T.contiguous()
        block.att.out_proj.bias.data = params[f"{prefix}attn.c_proj.bias"]

        # Feed-forward
        block.ff.layers[0].weight.data = params[f"{prefix}mlp.c_fc.weight"].T.contiguous()
        block.ff.layers[0].bias.data = params[f"{prefix}mlp.c_fc.bias"]
        block.ff.layers[2].weight.data = params[f"{prefix}mlp.c_proj.weight"].T.contiguous()
        block.ff.layers[2].bias.data = params[f"{prefix}mlp.c_proj.bias"]

    # Final layer norm
    model.final_norm.scale.data = params["ln_f.weight"]
    model.final_norm.shift.data = params["ln_f.bias"]

    # Output head (weight-tied with token embeddings in GPT-2)
    model.out_head.weight.data = params["wte.weight"]


def load_pretrained_gpt2(model_name="gpt2-small"):
    """Build GPTModel and load Hugging Face safetensors without TensorFlow."""
    if model_name not in GPT2_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(GPT2_MODELS)}")

    root = Path(__file__).resolve().parents[2]
    model_dir = root / "data" / "models" / GPT2_MODELS[model_name]["params"]
    saved_weights_path = model_dir / "model_weights.pth"
    config_path = model_dir / "config.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    if saved_weights_path.exists() and config_path.exists():
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        local_config = settings_to_local_config(settings)
        model = PreviousChapters.GPTModel(local_config)
        model.load_state_dict(
            torch.load(saved_weights_path, map_location="cpu", weights_only=True)
        )
    else:
        legacy_small_cache = root / "data" / "models" / "gpt2_hf"
        if model_name == "gpt2-small" and all(
            (legacy_small_cache / filename).exists()
            for filename in ("config.json", "model.safetensors")
        ):
            download_dir = legacy_small_cache
        else:
            download_dir = model_dir / "hf_download"
            download_gpt2_from_huggingface(download_dir, model_name=model_name)

        settings, params = load_gpt2_safetensors(download_dir)
        local_config = settings_to_local_config(settings)
        model = PreviousChapters.GPTModel(local_config)
        assign_gpt2_weights(model, params)
        config_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), saved_weights_path)

    model.eval()
    return model, local_config


if __name__ == "__main__":
    import argparse
    import torch
    import tiktoken
    import PreviousChapters

    parser = argparse.ArgumentParser(description="Load GPT-2 weights from Hugging Face")
    parser.add_argument(
        "--model", type=str, default="gpt2-small",
        choices=list(GPT2_MODELS.keys()),
        help="GPT-2 model variant to download (default: gpt2-small)"
    )
    args = parser.parse_args()
    model_name = args.model

    root = Path(__file__).resolve().parents[2]
    model_dir = root / "data" / "models" / GPT2_MODELS[model_name]["params"]
    model_dir.mkdir(parents=True, exist_ok=True)
    saved_weights_path = model_dir / "model_weights.pth"

    print(f"Using model: {model_name} ({GPT2_MODELS[model_name]['params']} parameters)")
    print(f"Model directory: {model_dir}")

    if saved_weights_path.exists():
        # Load directly from previously saved .pth file
        print(f"Found saved model at {saved_weights_path}, loading from disk...")
        config_path = model_dir / "config.json"
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        local_cfg = settings_to_local_config(settings)
        model = PreviousChapters.GPTModel(local_cfg)
        model.load_state_dict(torch.load(saved_weights_path, map_location="cpu", weights_only=True))
        model.eval()
        print(f"Model loaded from saved weights.")
    else:
        # Download from HF, assign weights, and save for future use
        hf_download_dir = model_dir / "hf_download"

        # Step 1: Download from Hugging Face
        download_gpt2_from_huggingface(hf_download_dir, model_name=model_name)

        # Step 2: Load weights and config
        settings, params = load_gpt2_safetensors(hf_download_dir)
        print("HF Config:", json.dumps(settings, indent=2)[:200], "...")
        print("Settings:", json.dumps(settings))

        print("Number of weight tensors:", len(params))
        #print("Parameter keys", params.keys())
        print(params["wte.weight"].shape, params["wpe.weight"].shape)
        print("Params weight dimensions", params["wte.weight"].shape)

        # Step 3: Build local GPTModel and assign weights
        local_cfg = settings_to_local_config(settings)
        print("Local config:", local_cfg)

        model = PreviousChapters.GPTModel(local_cfg)
        assign_gpt2_weights(model, params)
        model.eval()

        # Save config and model weights for future use
        config_dest = model_dir / "config.json"
        config_dest.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), saved_weights_path)
        print(f"Model weights saved to {saved_weights_path}")

    print(f"Model loaded with OpenAI {model_name} ({GPT2_MODELS[model_name]['params']}) weights.")
    print("Model Settings", json.dumps(settings, indent=2)[:200], "...")
    # Step 4: Quick generation test
    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "Every effort moves you"
    encoded = tokenizer.encode(start_text)
    idx = torch.tensor(encoded).unsqueeze(0)

    token_ids = PreviousChapters.generate_text_simple(
        model=model,
        idx=idx,
        max_new_tokens=25,
        context_size=local_cfg["context_length"],
    )

    output_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(f"\nInput:  {start_text}")
    print(f"Output: {output_text}")