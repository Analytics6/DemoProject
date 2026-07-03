from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FineTuneConfig:
    base_model: str
    learning_rate: float = 2e-4
    batch_size: int = 8
    epochs: int = 3
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit_quantization: bool = False
    target_modules: List[str] = None

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class ModelFineTune:
    """
    Strategy helper for model fine-tuning workflows.
    This file describes and tracks tuning approaches; integrate
    training frameworks (Transformers/PEFT/TRL) in production.
    """

    def lora(self, config: FineTuneConfig) -> Dict:
        return {
            "method": "LoRA",
            "base_model": config.base_model,
            "trainable_components": config.target_modules,
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "note": "Low-rank adapters inserted in attention layers.",
        }

    def qlora(self, config: FineTuneConfig) -> Dict:
        config.use_4bit_quantization = True
        base = self.lora(config)
        base.update(
            {
                "method": "QLoRA",
                "quantization": "4-bit",
                "note": "LoRA + 4-bit quantized base model for lower memory usage.",
            }
        )
        return base

    def peft(self, config: FineTuneConfig) -> Dict:
        return {
            "method": "PEFT",
            "base_model": config.base_model,
            "approach": "parameter_efficient_fine_tuning",
            "candidate_techniques": ["LoRA", "Prefix Tuning", "Prompt Tuning", "Adapters"],
            "recommended": "LoRA",
            "note": "Tune small parameter subsets while freezing base model.",
        }

    def rlhf(self, base_model: str) -> Dict:
        return {
            "method": "RLHF",
            "base_model": base_model,
            "stages": [
                "Collect preference pairs from human annotators",
                "Train reward model on preferences",
                "Optimize policy model with PPO/DPO-style loop",
                "Safety red-team evaluation and iteration",
            ],
            "note": "Aligns model behavior with user preferences and safety policy.",
        }

    def adapter_merge_quality(
        self, base_eval_score: float, adapter_eval_score: float, merged_eval_score: float
    ) -> Dict:
        delta_adapter = adapter_eval_score - base_eval_score
        delta_merge = merged_eval_score - base_eval_score
        retention = 0.0 if delta_adapter <= 0 else (delta_merge / delta_adapter)
        return {
            "base_eval_score": round(base_eval_score, 4),
            "adapter_eval_score": round(adapter_eval_score, 4),
            "merged_eval_score": round(merged_eval_score, 4),
            "adapter_gain": round(delta_adapter, 4),
            "merge_gain": round(delta_merge, 4),
            "adapter_merge_quality": round(retention, 4),
        }
