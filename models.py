"""The LLMs shared by vote and comparitor, and the vLLM settings each needs."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Model:
    name: str
    max_tokens_thinking: int  # room for the thought as well
    llm_args: dict = field(default_factory=dict)  # extra arguments for LLM()
    chat_args: dict = field(default_factory=dict)  # extra chat template arguments

    @property
    def gpus(self):
        # the GPUs that one replica of the model spans
        return self.llm_args.get("tensor_parallel_size", 1)


MODELS = {
    "fast": Model("google/gemma-4-E4B-it", 4096),
    "balanced": Model(
        "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",  # about 26 GB
        8192,
        # text only, so that no memory is reserved for the vision encoder
        llm_args={"limit_mm_per_prompt": {"image": 0, "audio": 0}},
    ),
    "accurate": Model(
        "cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4",  # calibrated AWQ, group 32
        16384,
        llm_args={
            "tensor_parallel_size": 4,
            "max_num_seqs": 64,
            # the n-gram table in pinned host RAM
            "engram_config": {"cpu_offload": True},
            "limit_mm_per_prompt": {"image": 0, "video": 0},
        },
        # the thinking effort (xhigh by default) is only a hint added to the system prompt, not a hard
        # limit, and is ignored when thinking is off
        chat_args={"reasoning_effort": "medium"},
    ),
}
MAX_MODEL_LEN = 65536