"""Pydantic response validation with type coercion and reasoning check."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.canonical_schema import ALL_Q_IDS, normalise_response
 
class SyntheticResponse:
    def __init__(self, synthetic_id, model, reasoning, responses):
        self.synthetic_id = synthetic_id
        self.model = model
        self.reasoning = reasoning
        self.responses = responses
 
    @classmethod
    def from_raw(cls, synthetic_id: str, model: str, parsed: dict):
        reasoning = str(parsed.get('reasoning', '')).strip()
        if len(reasoning) < 20:
            raise ValueError(f'reasoning too short: {repr(reasoning)}')
        if not any(c.isalpha() for c in reasoning):
            raise ValueError(f'reasoning has no words: {repr(reasoning)}')
 
        responses, missing, bad = {}, [], []
        for qid in ALL_Q_IDS:
            raw = parsed.get(qid)
            if raw is None: missing.append(qid); continue
            coerced = normalise_response(raw)
            if coerced not in {'1','2','3','4','5'}: bad.append((qid, raw)); continue
            responses[qid] = int(coerced)
 
        if missing: raise ValueError(f'Missing question IDs: {missing}')
        if bad:     raise ValueError(f'Invalid response values: {bad}')
        return cls(synthetic_id, model, reasoning, responses)
 
    def to_row(self) -> dict:
        row = {'synthetic_id': self.synthetic_id, 'model': self.model,
               'reasoning': self.reasoning}
        row.update(self.responses)
        return row
