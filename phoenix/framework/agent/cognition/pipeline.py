import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional


BrainFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass(frozen=True)
class PipelineStep:
    id: str
    brain: str
    input_map: Dict[str, str]
    output_key: str
    required_inputs: List[str]


class PipelineValidationError(ValueError):
    pass


class BrainRegistry:
    """
    Runtime registry for cognition brain handlers.
    Enables plugging new brains (e.g. generator) into JSON pipelines.
    """

    def __init__(self):
        self._handlers: Dict[str, BrainFn] = {}

    def register(self, name: str, handler: BrainFn) -> None:
        if not name or not isinstance(name, str):
            raise PipelineValidationError("Brain name must be a non-empty string.")
        if not callable(handler):
            raise PipelineValidationError(f"Brain handler for '{name}' must be callable.")
        self._handlers[name] = handler

    def get(self, name: str) -> BrainFn:
        if name not in self._handlers:
            raise PipelineValidationError(f"Brain '{name}' is not registered in pipeline runtime.")
        return self._handlers[name]


class CognitionPipeline:
    """
    Strict JSON-defined pipeline executor.
    Each step maps explicit state keys into a brain handler and writes output to one key.
    """

    def __init__(self, spec: Dict[str, Any]):
        self.name = str(spec.get("name", "unnamed_pipeline"))
        self.version = str(spec.get("version", "1.0"))
        self.steps = self._parse_steps(spec)

    @classmethod
    def from_json_file(cls, path: str) -> "CognitionPipeline":
        raw = Path(path).read_text(encoding="utf-8")
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as e:
            raise PipelineValidationError(f"Invalid JSON pipeline spec '{path}': {e}") from e
        return cls(spec)

    def _parse_steps(self, spec: Dict[str, Any]) -> List[PipelineStep]:
        raw_steps = spec.get("steps", [])
        if not isinstance(raw_steps, list):
            raise PipelineValidationError("Pipeline spec must include 'steps' array.")

        seen_ids = set()
        parsed: List[PipelineStep] = []
        for raw in raw_steps:
            if not isinstance(raw, dict):
                raise PipelineValidationError("Each pipeline step must be an object.")

            step = PipelineStep(
                id=str(raw.get("id", "")).strip(),
                brain=str(raw.get("brain", "")).strip(),
                input_map=raw.get("input_map", {}) or {},
                output_key=str(raw.get("output_key", "")).strip(),
                required_inputs=raw.get("required_inputs", []) or [],
            )

            if not step.id:
                raise PipelineValidationError("Pipeline step requires non-empty 'id'.")
            if step.id in seen_ids:
                raise PipelineValidationError(f"Duplicate pipeline step id '{step.id}'.")
            seen_ids.add(step.id)

            if not step.brain:
                raise PipelineValidationError(f"Pipeline step '{step.id}' requires non-empty 'brain'.")
            if not isinstance(step.input_map, dict):
                raise PipelineValidationError(f"Pipeline step '{step.id}' input_map must be an object.")
            if not step.output_key:
                raise PipelineValidationError(f"Pipeline step '{step.id}' requires non-empty 'output_key'.")
            if not isinstance(step.required_inputs, list):
                raise PipelineValidationError(f"Pipeline step '{step.id}' required_inputs must be an array.")

            parsed.append(step)
        return parsed

    async def run(
        self,
        registry: BrainRegistry,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(initial_state or {})

        for step in self.steps:
            for key in step.required_inputs:
                if key not in state:
                    raise PipelineValidationError(
                        f"Step '{step.id}' missing required input '{key}'. State keys: {sorted(state.keys())}"
                    )

            handler = registry.get(step.brain)
            payload = {
                payload_key: state[state_key]
                for payload_key, state_key in step.input_map.items()
                if state_key in state
            }
            output = await handler(payload)
            if not isinstance(output, dict):
                raise PipelineValidationError(
                    f"Step '{step.id}' brain '{step.brain}' must return dict, got {type(output).__name__}."
                )
            state[step.output_key] = output

        return state
