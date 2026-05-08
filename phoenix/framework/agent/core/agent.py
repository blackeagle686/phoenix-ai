import uuid
from typing import Any, Callable, Dict, Optional, Type
import inspect
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.memory.hybrid import HybridMemory
from phoenix.framework.agent.memory.adapter import InteractiveMemoryAdapter
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.actor import Actor
from phoenix.framework.agent.execution.tool_manager import ToolManager
from phoenix.framework.agent.core.loop import AgentLoop
from phoenix.framework.agent.core.profile import AgentProfile
from typing import Union

class Agent:
    """The main Agent class that integrates LLM, memory, tools, and cognition modules."""
    def __init__(
        self,
        llm=None,
        memory=None,
        tools=None,
        thinker=None,
        planner=None,
        analyzer=None,
        actor=None,
        reflector=None,
        tool_manager=None,
        loop=None,
        profile: Optional[Union[AgentProfile, str, dict]] = None,
        loop_cls: Optional[Type[AgentLoop]] = None,
        component_factories: Optional[Dict[str, Callable[..., Any]]] = None,
    ):
        # Default LLM to OpenAILLM (using LongCat-Flash-Chat by default via its config fallback)
        self.llm = llm if llm is not None else OpenAILLM()
        self.memory = self._prepare_memory(memory)
        self.tools = tools if tools is not None else ToolRegistry.load_default()
        self._factories = component_factories or {}
        self._dynamic_components: Dict[str, Any] = {}
        self.loop_cls = loop_cls or AgentLoop
        
        self.profile = self._prepare_profile(profile)

        # Setup framework with explicit override -> factory -> default precedence
        self.tool_manager = tool_manager or self._build_component("tool_manager")
        if self.tool_manager is None:
            self.tool_manager = ToolManager(self.tools)

        self.thinker = thinker or self._build_component("thinker")
        if self.thinker is None:
            self.thinker = Thinker(self.llm, profile=self.profile)

        self.planner = planner or self._build_component("planner")
        if self.planner is None:
            self.planner = Planner(self.llm, self.tools, profile=self.profile)

        self.actor = actor or self._build_component("actor")
        if self.actor is None:
            self.actor = Actor(self.tool_manager)

        self.reflector = reflector or self._build_component("reflector")
        if self.reflector is None:
            self.reflector = Reflector(self.llm, profile=self.profile)

        self.analyzer = analyzer or self._build_component("analyzer")
        if self.analyzer is None:
            self.analyzer = Analyzer(self.llm, profile=self.profile)

        # Build user-defined custom components (e.g. generator) from factories.
        self._build_dynamic_components()

        self.loop = loop or self._build_component("loop")
        if self.loop is None:
            self.loop = self._construct_loop()

    def _prepare_memory(self, memory):
        if memory is None:
            return HybridMemory()

        required = ("add_interaction", "get_full_context")
        has_required = all(hasattr(memory, name) for name in required)
        has_layers = hasattr(memory, "session") and hasattr(memory, "reflection")

        if has_required and has_layers:
            return memory

        return InteractiveMemoryAdapter(memory)

    def _prepare_profile(self, profile: Optional[Union[AgentProfile, str, dict]]) -> Optional[AgentProfile]:
        if profile is None:
            return None
        if isinstance(profile, AgentProfile):
            return profile
        if isinstance(profile, str):
            return AgentProfile.from_json(profile)
        if isinstance(profile, dict):
            return AgentProfile.from_dict(profile)
        raise ValueError("Profile must be an AgentProfile instance, dict, or file path string.")

    def _build_component(self, name: str):
        """
        Optional construction hook for framework extensibility.
        Factory signature can accept any subset of:
        llm, memory, tools, thinker, planner, analyzer, actor, reflector, tool_manager,
        loop_cls, profile, plus any custom components built from other factories.
        """
        factory = self._factories.get(name)
        if not callable(factory):
            return None
        return self._call_with_supported_kwargs(factory, self._component_context())

    def _component_context(self) -> Dict[str, Any]:
        context = {
            "llm": self.llm,
            "memory": self.memory,
            "tools": self.tools,
            "thinker": getattr(self, "thinker", None),
            "planner": getattr(self, "planner", None),
            "analyzer": getattr(self, "analyzer", None),
            "actor": getattr(self, "actor", None),
            "reflector": getattr(self, "reflector", None),
            "tool_manager": getattr(self, "tool_manager", None),
            "loop_cls": self.loop_cls,
            "profile": self.profile,
        }
        context.update(self._dynamic_components)
        return context

    def _build_dynamic_components(self):
        reserved = {"tool_manager", "thinker", "planner", "analyzer", "actor", "reflector", "loop"}
        for name in self._factories.keys():
            if name in reserved:
                continue
            if hasattr(self, name) and getattr(self, name) is not None:
                self._dynamic_components[name] = getattr(self, name)
                continue
            component = self._build_component(name)
            if component is not None:
                setattr(self, name, component)
                self._dynamic_components[name] = component

    def _call_with_supported_kwargs(self, fn: Callable[..., Any], kwargs: Dict[str, Any]):
        sig = inspect.signature(fn)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_var_kw:
            return fn(**kwargs)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**supported)

    def _construct_loop(self):
        loop_kwargs = {
            "thinker": self.thinker,
            "planner": self.planner,
            "actor": self.actor,
            "reflector": self.reflector,
            "analyzer": self.analyzer,
        }
        loop_kwargs.update(self._dynamic_components)
        return self._call_with_supported_kwargs(self.loop_cls, loop_kwargs)

    def register_tool(self, tool):
        """Helper to register a new tool to the agent's ToolRegistry."""
        self.tools.register(tool)

    def set_component(self, name: str, component: Any, rebuild_loop: bool = True):
        """
        Replace a core component at runtime.
        For core names, assigns explicitly.
        Any other name is treated as a user-defined custom component.
        """
        setattr(self, name, component)
        if name not in {"thinker", "planner", "analyzer", "actor", "reflector", "tool_manager", "loop"}:
            self._dynamic_components[name] = component
        if rebuild_loop and name != "loop":
            self.rebuild_loop()

    def rebuild_loop(self, loop_cls: Optional[Type[AgentLoop]] = None):
        """
        Rebuild loop from current components.
        Useful after swapping thinker/planner/analyzer/actor/reflector.
        """
        if loop_cls is not None:
            self.loop_cls = loop_cls
        self.loop = self._construct_loop()

    def register_brain(self, name: str, handler: Callable[..., Any]):
        """
        Register a custom cognition brain handler in loop pipeline runtime.
        """
        if not hasattr(self.loop, "register_brain"):
            raise ValueError("Current loop does not support custom brain registration.")
        self.loop.register_brain(name, handler)

    def set_cognition_pipeline(self, bootstrap_pipeline_path: Optional[str] = None, iteration_pipeline_path: Optional[str] = None):
        """
        Replace default cognition JSON pipelines with custom specs.
        """
        if not hasattr(self.loop, "set_pipeline_specs"):
            raise ValueError("Current loop does not support pipeline specs.")
        self.loop.set_pipeline_specs(
            bootstrap_pipeline_path=bootstrap_pipeline_path,
            iteration_pipeline_path=iteration_pipeline_path
        )

    async def run(self, prompt: str, session_id: str = None, max_iterations: int = 15, mode: str = "auto") -> str:
        """
        Run the agent with the given prompt.
        mode can be "auto", "plan", or "fast_ans".
        """
        # Auto-initialize LLM if it's our OpenAILLM and hasn't been initialized
        if hasattr(self.llm, "client") and self.llm.client is None:
            if hasattr(self.llm, "init"):
                await self.llm.init()

        if session_id is None:
            session_id = str(uuid.uuid4())
            
        # Add user prompt to memory
        await self.memory.add_interaction(session_id, "user", prompt)
        
        # Decide mode if auto
        actual_mode = mode
        if mode == "auto":
            classification_prompt = (
                f"Analyze the following user prompt and decide if it requires using tools and multi-step planning (like writing files, searching the web, or complex logic), "
                f"or if it is a simple question that can be answered immediately.\n\n"
                f"Prompt: {prompt}\n\n"
                f"Respond with exactly one word: 'PLAN' or 'FAST'."
            )
            classification = await self.llm.generate(classification_prompt, session_id=None, max_tokens=10)
            if "PLAN" in classification.upper():
                actual_mode = "plan"
            else:
                actual_mode = "fast_ans"
                
        if actual_mode == "fast_ans":
            # Fast answer mode: bypass the agent loop and just use the LLM directly with memory context
            context = await self.memory.get_full_context(session_id, query=prompt)
            fast_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAnswer directly:"
            fast_answer = await self.llm.generate(fast_prompt, session_id=None)
            await self.memory.add_interaction(session_id, "assistant", fast_answer)
            return fast_answer
        else:
            # Start execution loop
            return await self.loop.run(prompt, self.memory, session_id, max_iterations=max_iterations)

    async def run_stream(self, prompt: str, session_id: str = None, max_iterations: int = 15, mode: str = "auto"):
        """
        Streaming variant of run(). Uses loop.run_stream when available.
        """
        if hasattr(self.llm, "client") and self.llm.client is None:
            if hasattr(self.llm, "init"):
                await self.llm.init()

        if session_id is None:
            session_id = str(uuid.uuid4())

        await self.memory.add_interaction(session_id, "user", prompt)

        actual_mode = mode
        if mode == "auto":
            classification_prompt = (
                f"Analyze the following user prompt and decide if it requires using tools and multi-step planning (like writing files, searching the web, or complex logic), "
                f"or if it is a simple question that can be answered immediately.\n\n"
                f"Prompt: {prompt}\n\n"
                f"Respond with exactly one word: 'PLAN' or 'FAST'."
            )
            classification = await self.llm.generate(classification_prompt, session_id=None, max_tokens=10)
            actual_mode = "plan" if "PLAN" in classification.upper() else "fast_ans"

        if actual_mode == "fast_ans":
            context = await self.memory.get_full_context(session_id, query=prompt)
            fast_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAnswer directly:"
            fast_answer = await self.llm.generate(fast_prompt, session_id=None)
            await self.memory.add_interaction(session_id, "assistant", fast_answer)
            yield {"type": "status", "content": "Fast answer mode"}
            for line in fast_answer.splitlines() or [fast_answer]:
                yield {"type": "chunk", "content": f"{line}\n"}
            return

        if hasattr(self.loop, "run_stream"):
            async for event in self.loop.run_stream(prompt, self.memory, session_id, max_iterations=max_iterations):
                yield event
            return

        final = await self.loop.run(prompt, self.memory, session_id, max_iterations=max_iterations)
        yield {"type": "status", "content": "Preparing final response..."}
        for line in final.splitlines() or [final]:
            yield {"type": "chunk", "content": f"{line}\n"}
