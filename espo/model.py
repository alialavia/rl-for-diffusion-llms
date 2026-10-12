"""Loading LLaDA / Dream with optional LoRA, wrapped in the trainer's Policy interface."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Optional

import torch

# LLaDA's linear layers are named q_proj/k_proj/v_proj/attn_out/ff_proj/up_proj/ff_out.
# The authors' config lists Llama names; only q/k/v/up_proj exist in LLaDA, so that is
# the set their runs actually adapted. We default to the same effective set.
LLADA_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "up_proj"]
DREAM_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


@dataclass
class ModelArgs:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    family: str = "auto"                 # "llada" | "dream" | "auto"
    torch_dtype: str = "bfloat16"
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 64
    lora_dropout: float = 0.0            # 0 keeps L_theta and L_old the same function
    lora_target_modules: Optional[list] = None
    gradient_checkpointing: bool = False
    adapter_path: Optional[str] = None   # resume / evaluate a trained adapter
    mask_token_id: Optional[int] = None


def detect_family(name: str) -> str:
    n = name.lower()
    if "dream" in n:
        return "dream"
    return "llada"


class HFPolicy:
    """Policy interface over a HF diffusion LM (optionally PEFT + DDP wrapped)."""

    def __init__(self, model, family: str, mask_id: int, ddp_model=None, ref_model=None):
        self.model = model          # unwrapped (PeftModel or PreTrainedModel)
        self.ddp = ddp_model        # DistributedDataParallel(model) or None
        self.family = family
        self.mask_id = mask_id
        self.ref_model = ref_model  # only used for full fine-tuning
        self.is_peft = hasattr(model, "disable_adapter")

    def _forward(self, module, ids, attn):
        if self.family == "dream":
            if attn is not None and bool((attn == 0).any()):
                pos = attn.long().cumsum(-1) - 1
                pos = pos.masked_fill(attn == 0, 1)
                a = attn.bool()
                mask4d = a[:, None, None, :] & a[:, None, :, None]
                logits = module(ids, mask4d, pos).logits
            else:
                logits = module(ids, "full", None).logits
            # Dream predicts token i from position i-1
            return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        if attn is not None and not bool((attn == 0).any()):
            attn = None
        return module(input_ids=ids, attention_mask=attn).logits

    def logits(self, ids, attn=None):
        return self._forward(self.ddp if self.ddp is not None else self.model, ids, attn)

    def generation_logits(self, ids, attn=None):
        return self._forward(self.model, ids, attn)

    @contextmanager
    def reference_logits(self):
        if self.is_peft:
            with self.model.disable_adapter():
                yield lambda ids, attn=None: self._forward(self.model, ids, attn)
        else:
            if self.ref_model is None:
                raise RuntimeError("full fine-tuning with beta > 0 needs a reference model")
            yield lambda ids, attn=None: self._forward(self.ref_model, ids, attn)

    def no_sync(self):
        return self.ddp.no_sync() if self.ddp is not None else nullcontext()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train(self):
        self.model.train()
        if self.ref_model is not None:
            self.ref_model.train()

    def eval(self):
        self.model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()


def load_model_and_tokenizer(args: ModelArgs, device, trainable: bool = True, need_ref: bool = False):
    from transformers import AutoModel, AutoTokenizer

    family = detect_family(args.model_name_or_path) if args.family == "auto" else args.family
    dtype = getattr(torch, args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def load():
        m = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=dtype)
        return m.to(device)

    model = load()
    mask_id = args.mask_token_id or getattr(model.config, "mask_token_id", None) or (
        126336 if family == "llada" else 151666)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model, family)

    ref_model = None
    if args.use_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if args.adapter_path:
            model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=trainable)
        elif trainable:
            targets = args.lora_target_modules or (
                LLADA_LORA_TARGETS if family == "llada" else DREAM_LORA_TARGETS)
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules=targets, bias="none"))
    elif trainable and need_ref:
        ref_model = load().eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tok, family, mask_id, ref_model


def enable_gradient_checkpointing(model, family: str):
    inner = getattr(model, "model", None)
    if family == "llada" and inner is not None and hasattr(inner, "set_activation_checkpointing"):
        inner.set_activation_checkpointing("whole_layer")
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
