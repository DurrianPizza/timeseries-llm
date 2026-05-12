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

    Args:
        llm_name: HuggingFace or ModelScope model name for Qwen
        encoder_dim: Hidden dim of TimeSeries encoder
        llm_dim: Hidden dim of LLM
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen3.5-0.8B",
        encoder_dim: int = 256,
        llm_dim: int = 896,
        device: str = "cpu",
    ):
        super().__init__()
        self.llm_name = llm_name
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        # TimeSeries Encoder - linear projection without CNN/Transformer
        from timeseries_llm.models.encoder import TimeSeriesEncoder
        self.encoder = TimeSeriesEncoder(
            in_channels=8,  # max_dims from config
            hidden_dim=encoder_dim,
        )

        # Fusion MLP
        from timeseries_llm.models.fusion import MLPFusion
        self.fusion = MLPFusion(encoder_dim=encoder_dim, llm_dim=llm_dim)

        # LLM - try HuggingFace first, then ModelScope
        # Note: MPS doesn't support bf16 well, so we pass device to handle dtype
        self.llm, self.tokenizer = _load_model_and_tokenizer(llm_name, device=device)

        # Move encoder and fusion to the same device and dtype as LLM
        self.encoder = self.encoder.to(device=device, dtype=self.llm.dtype)
        self.fusion = self.fusion.to(device=device, dtype=self.llm.dtype)

    def forward(self, input_ids: torch.LongTensor, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            input_ids: Token IDs for text input, shape (batch, text_seq_len)
            encoder_outputs: TimeSeries encoded tensor, shape (batch, ts_seq_len, encoder_dim)
            attention_mask: Attention mask for text tokens only (will be extended for time series)

        Returns:
            LLM output logits
        """
        # Get LLM embeddings
        text_embeddings = self.llm.model.embed_tokens(input_ids)
        text_seq_len = text_embeddings.shape[1]

        # Concatenate: text + time series (after fusion projection)
        # Fusion runs in fp32 for numerical stability, then convert output to model dtype
        model_dtype = self.llm.dtype
        fused_encoder_outputs = self.fusion(encoder_outputs)
        ts_seq_len = fused_encoder_outputs.shape[1]

        # Ensure embeddings match model dtype (model may be bf16)
        combined_embeddings = torch.cat([text_embeddings.to(model_dtype), fused_encoder_outputs.to(model_dtype)], dim=1)

        # Extend attention mask to cover both text and time series tokens
        # Time series tokens can attend to all time series tokens (full attention within ts)
        # and can attend to all text tokens
        if attention_mask is not None:
            # attention_mask shape: (batch, text_seq_len)
            # Extend with ones for time series tokens: (batch, ts_seq_len)
            ts_attention_mask = attention_mask.new_ones(attention_mask.shape[0], ts_seq_len)
            extended_attention_mask = torch.cat([attention_mask, ts_attention_mask], dim=1)
        else:
            extended_attention_mask = None

        # Forward through LLM - don't pass labels, let trainer compute loss
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=extended_attention_mask,
        )
        return outputs.logits

    def generate(self, input_ids: torch.LongTensor, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor = None, max_new_tokens: int = 100) -> torch.LongTensor:
        """Generate text given time series and question.

        Args:
            input_ids: Token IDs for text input
            encoder_outputs: TimeSeries encoded tensor
            attention_mask: Attention mask for text tokens only
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated token IDs
        """
        text_embeddings = self.llm.model.embed_tokens(input_ids)
        text_seq_len = text_embeddings.shape[1]

        fused_encoder_outputs = self.fusion(encoder_outputs)
        ts_seq_len = fused_encoder_outputs.shape[1]

        # Ensure embeddings match model dtype
        model_dtype = self.llm.dtype
        combined_embeddings = torch.cat([text_embeddings.to(model_dtype), fused_encoder_outputs.to(model_dtype)], dim=1)

        # Extend attention mask for time series tokens
        if attention_mask is not None:
            ts_attention_mask = attention_mask.new_ones(attention_mask.shape[0], ts_seq_len)
            extended_attention_mask = torch.cat([attention_mask, ts_attention_mask], dim=1)
        else:
            extended_attention_mask = None

        outputs = self.llm.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
        return outputs