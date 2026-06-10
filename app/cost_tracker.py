PRICING = {
    "gpt-4o-mini": {
        "prompt": 0.00015 / 1000,
        "completion": 0.00060 / 1000,
    },
    "gpt-3.5-turbo": {
        "prompt": 0.00015 / 1000,
        "completion": 0.00060 / 1000,
    },
    "text-embedding-3-small": {
        "prompt": 0.00002 / 1000,
        "completion": 0.0,
    },
    "text-embedding-3-large": {
        "prompt": 0.00013 / 1000,
        "completion": 0.0,
    },
}

DEFAULT_PRICING = {"prompt": 0.0, "completion": 0.0}


class RequestCostTracker:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.calls: list[dict] = []

    def add_call(self, model: str, prompt_tokens: int, completion_tokens: int, call_type: str) -> None:
        rates = PRICING.get(model, DEFAULT_PRICING)
        cost_usd = prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]
        self.calls.append({
            "call_type": call_type,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        })

    def summary(self) -> dict:
        total_tokens = sum(call["prompt_tokens"] + call["completion_tokens"] for call in self.calls)
        total_cost_usd = sum(call["cost_usd"] for call in self.calls)

        breakdown_by_model: dict[str, dict] = {}
        for call in self.calls:
            entry = breakdown_by_model.setdefault(call["model"], {"tokens": 0, "cost_usd": 0.0})
            entry["tokens"] += call["prompt_tokens"] + call["completion_tokens"]
            entry["cost_usd"] += call["cost_usd"]

        return {
            "request_id": self.request_id,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "calls": self.calls,
            "breakdown_by_model": breakdown_by_model,
        }
