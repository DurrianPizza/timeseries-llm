import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class TimeSeriesLLM(nn.Module):
    """TimeSeries-LLM wrapper combining encoder + fusion + LLM.

    Args:
        llm_name: HuggingFace model name for Qwen
        encoder_dim: Hidden dim of TimeSeries encoder
        llm_dim: Hidden dim of LLM
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2-0.5B-Instruct",
        encoder_dim: int = 256,
        llm_dim: int = 896,
        num_encoder_layers: int = 2,
    ):
        super().__init__()
        self.llm_name = llm_name
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        # TimeSeries Encoder
        from timeseries_llm.models.encoder import TimeSeriesEncoder
        self.encoder = TimeSeriesEncoder(
            in_channels=1,  # will be overridden based on input
            hidden_dim=encoder_dim,
            num_layers=num_encoder_layers,
        )

        # Fusion MLP
        from timeseries_llm.models.fusion import MLPFusion
        self.fusion = MLPFusion(encoder_dim=encoder_dim, llm_dim=llm_dim)

        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

    def forward(self, input_ids: torch.LongTensor, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.LongTensor = None):
        """
        Args:
            input_ids: Token IDs for text input, shape (batch, text_seq_len)
            encoder_outputs: TimeSeries encoded tensor, shape (batch, ts_seq_len, encoder_dim)
            attention_mask: Attention mask for combined sequence
            labels: Labels for language modeling loss

        Returns:
            LLM output with loss
        """
        # Get LLM embeddings
        text_embeddings = self.llm.model.embed_tokens(input_ids)

        # Concatenate: text + time series (after fusion projection)
        fused_encoder_outputs = self.fusion(encoder_outputs)
        combined_embeddings = torch.cat([text_embeddings, fused_encoder_outputs], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def generate(self, input_ids: torch.LongTensor, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor = None, max_new_tokens: int = 100) -> torch.LongTensor:
        """Generate text given time series and question.

        Args:
            input_ids: Token IDs for text input
            encoder_outputs: TimeSeries encoded tensor
            attention_mask: Attention mask for combined sequence
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated token IDs
        """
        text_embeddings = self.llm.model.embed_tokens(input_ids)
        fused_encoder_outputs = self.fusion(encoder_outputs)
        combined_embeddings = torch.cat([text_embeddings, fused_encoder_outputs], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
        return outputs