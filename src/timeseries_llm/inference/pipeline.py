import torch
from typing import Dict, List, Union
from timeseries_llm.models.llm_wrapper import TimeSeriesLLM


class TimeSeriesPipeline:
    """Inference pipeline for TimeSeries-LLM."""

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2-0.5B-Instruct",
        encoder_dim: int = 256,
        llm_dim: int = 896,
        checkpoint_path: str = None,
    ):
        print(f"[DEBUG] Pipeline init: encoder_dim={encoder_dim}, llm_dim={llm_dim}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeSeriesLLM(
            llm_name=llm_name,
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
        )
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        time_series: torch.Tensor,
        question: str,
        max_new_tokens: int = 100,
    ) -> str:
        """Answer a question about a time series."""
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(0)
        time_series = time_series.to(self.device)
        encoder_output = self.model.encoder(time_series)
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.model.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            encoder_outputs=encoder_output,
            max_new_tokens=max_new_tokens,
        )
        answer = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def batch_predict(
        self,
        time_series_list: List[torch.Tensor],
        questions: List[str],
        max_new_tokens: int = 100,
    ) -> List[str]:
        """Batch prediction for multiple time series and questions."""
        answers = []
        for ts, q in zip(time_series_list, questions):
            answer = self.predict(ts, q, max_new_tokens)
            answers.append(answer)
        return answers
