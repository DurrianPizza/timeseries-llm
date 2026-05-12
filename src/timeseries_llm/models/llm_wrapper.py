import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_model_and_tokenizer(model_name: str, device: str = "cpu"):
    """Load model and tokenizer from ModelScope or HuggingFace.

    Args:
        model_name: Model identifier (HuggingFace path or ModelScope path)
        device: Device to load model on (cpu, mps, cuda)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"[INFO] Loading model: {model_name}")

    # For MPS, use fp32 to avoid bf16 issues
    dtype = torch.float32 if device == "mps" else torch.bfloat16

    # Try ModelScope first (better for China)
    try:
        print(f"[INFO] Trying ModelScope...")
        from modelscope import AutoModelForCausalLM as MsModel
        from modelscope import AutoTokenizer as MsTokenizer
        print(f"[INFO] Downloading model from ModelScope (this may take a while)...")
        model = MsModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"[INFO] Model downloaded, loading tokenizer...")
        tokenizer = MsTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"[INFO] Model and tokenizer loaded successfully")
        # Force dtype conversion after load (dtype param may not work correctly on MPS)
        if device == "mps":
            model = model.to(dtype=torch.float32, device=device)
        else:
            model = model.to(device=device)
        return model, tokenizer
    except Exception as e:
        print(f"[INFO] ModelScope failed: {e}")

    # Fall back to HuggingFace
    try:
        print(f"[INFO] Trying HuggingFace...")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        if device == "mps":
            model = model.to(dtype=torch.float32, device=device)
        else:
            model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"[INFO] Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        raise ImportError(
            f"Failed to load model '{model_name}' from both HuggingFace and ModelScope. "
            f"Please check the model name and your network connection. Last error: {e}"
        )


class TimeSeriesLLM(nn.Module):
    """TimeSeries-LLM wrapper combining encoder + fusion + LLM.

    Supports two modes:
    - mode="direct": Direct projection of time series values to LLM dim (Plan A)
    - mode="encoder_decoder": Encoder with reconstruction decoder (Plan B)

    Args:
        llm_name: HuggingFace or ModelScope model name for Qwen
        encoder_dim: Hidden dim of TimeSeries encoder
        llm_dim: Hidden dim of LLM
        mode: "direct" or "encoder_decoder"
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3.5-0.8B",
        encoder_dim: int = 256,
        llm_dim: int = 896,
        device: str = "cpu",
        mode: str = "direct",
    ):
        super().__init__()
        self.llm_name = llm_name
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.mode = mode

        if mode == "direct":
            # Plan A: Direct projection - no compression, LLM sees raw values
            from timeseries_llm.models.timeseries_encoder import TimeSeriesEncoderDirect
            self.encoder = TimeSeriesEncoderDirect(
                in_channels=1,
                llm_dim=llm_dim,
            )
            self.has_decoder = False
        elif mode == "encoder_decoder":
            # Plan B: Encoder with reconstruction decoder
            from timeseries_llm.models.timeseries_encoder import TimeSeriesEncoderWithReconstruction
            self.encoder = TimeSeriesEncoderWithReconstruction(
                in_channels=1,
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
            )
            self.has_decoder = True
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'direct' or 'encoder_decoder'")

        # LLM - try HuggingFace first, then ModelScope
        self.llm, self.tokenizer = _load_model_and_tokenizer(llm_name, device=device)

        # Move encoder to the same device and dtype as LLM
        self.encoder = self.encoder.to(device=device, dtype=self.llm.dtype)

    def forward(self, input_ids: torch.LongTensor, ts_embeddings: torch.Tensor = None, attention_mask: torch.Tensor = None, raw_ts: torch.Tensor = None):
        """
        Args:
            input_ids: Token IDs for text input, shape (batch, text_seq_len)
            ts_embeddings: Pre-computed time series embeddings (for backward compat)
                           For mode="direct": shape (batch, seq_len, llm_dim)
                           For mode="encoder_decoder": shape (batch, seq_len, llm_dim)
            attention_mask: Attention mask for text tokens only (will be extended for time series)
            raw_ts: Raw time series tensor (batch, channels, seq_len) - used if ts_embeddings not provided

        Returns:
            For mode="direct": LLM output logits
            For mode="encoder_decoder": tuple of (LLM logits, reconstructed_ts)
        """
        # Get LLM embeddings
        text_embeddings = self.llm.model.embed_tokens(input_ids)

        # Get time series embeddings
        if ts_embeddings is None and raw_ts is not None:
            # Compute time series embeddings from raw input
            if self.mode == "direct":
                ts_embeddings = self.encoder(raw_ts)  # Already projected to llm_dim
            elif self.mode == "encoder_decoder":
                ts_embeddings, reconstructed = self.encoder(raw_ts)  # Returns (llm_out, reconstructed)
        elif ts_embeddings is None:
            raise ValueError("Must provide either ts_embeddings or raw_ts")

        ts_seq_len = ts_embeddings.shape[1]

        # Put time series BEFORE text - critical so causal attention allows text to attend to ts
        combined_embeddings = torch.cat([ts_embeddings, text_embeddings], dim=1)

        # Extend attention mask: time series tokens first, then text tokens
        if attention_mask is not None:
            ts_attention_mask = attention_mask.new_ones(attention_mask.shape[0], ts_seq_len)
            extended_attention_mask = torch.cat([ts_attention_mask, attention_mask], dim=1)
        else:
            extended_attention_mask = None

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=extended_attention_mask,
        )

        if self.mode == "encoder_decoder":
            return outputs.logits, reconstructed
        return outputs.logits

    def generate(self, input_ids: torch.LongTensor, ts_embeddings: torch.Tensor = None, attention_mask: torch.Tensor = None, raw_ts: torch.Tensor = None, max_new_tokens: int = 100, eos_token_id: int = None) -> torch.LongTensor:
        """Generate text given time series and question.

        Args:
            input_ids: Token IDs for text input
            ts_embeddings: Pre-computed time series embeddings
            attention_mask: Attention mask for text tokens only
            raw_ts: Raw time series tensor (used if ts_embeddings not provided)
            max_new_tokens: Maximum tokens to generate
            eos_token_id: End-of-sequence token id

        Returns:
            Generated token IDs
        """
        text_embeddings = self.llm.model.embed_tokens(input_ids)

        # Get time series embeddings
        if ts_embeddings is None and raw_ts is not None:
            if self.mode == "direct":
                ts_embeddings = self.encoder(raw_ts)
            elif self.mode == "encoder_decoder":
                ts_embeddings, _ = self.encoder(raw_ts)
        elif ts_embeddings is None:
            raise ValueError("Must provide either ts_embeddings or raw_ts")

        ts_seq_len = ts_embeddings.shape[1]

        # Put time series BEFORE text (same as forward)
        combined_embeddings = torch.cat([ts_embeddings, text_embeddings], dim=1)

        # Extend attention mask: time series first, then text
        if attention_mask is not None:
            ts_attention_mask = attention_mask.new_ones(attention_mask.shape[0], ts_seq_len)
            extended_attention_mask = torch.cat([ts_attention_mask, attention_mask], dim=1)
        else:
            extended_attention_mask = None

        generate_kwargs = {
            "inputs_embeds": combined_embeddings,
            "attention_mask": extended_attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
        }
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        outputs = self.llm.generate(**generate_kwargs)
        return outputs