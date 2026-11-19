import uuid
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union
from phoenix.services.llm.openai import OpenAILLM
from phoenix.services.llm.local import LocalLLM
from phoenix.services.llm.router import LLMRouter
from phoenix.services.observability.tracing import tracer
from phoenix.framework.agent.memory.manager import MemoryManager
from phoenix.framework.agent.memory.adapter import InteractiveMemoryAdapter
from phoenix.framework.agent.tools.base import BaseTool, create_tool
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.cognition.thinker import Thinker
from phoenix.framework.agent.cognition.planner import Planner
from phoenix.framework.agent.cognition.reflector import Reflector
from phoenix.framework.agent.cognition.analyzer import Analyzer
from phoenix.framework.agent.cognition.actor import Actor
from phoenix.framework.agent.execution.tool_manager import ToolManager
from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
from phoenix.framework.agent.core.loop import AgentLoop
from phoenix.framework.agent.core.profile import AgentProfile


class Agent:
    """The main Agent class that integrates LLM(s), memory, tools, runtime, telemetry, and cognition modules."""
    def __init__(
        self,
        llm=None,
        llms: Optional[Iterable[Any]] = None,
        fallback_llm: Optional[Any] = None,
        fallback_llms: Optional[Iterable[Any]] = None,
        memory=None,
        tools=None,
        custom_tools: Optional[Iterable[Any]] = None,
        include_default_tools: bool = False,
        thinker=None,
        planner=None,
        analyzer=None,
        actor=None,
        reflector=None,
        tool_manager=None,
        runtime=None,
        loop=None,
        profile: Optional[Union[AgentProfile, str, dict]] = None,
        loop_cls: Optional[Type[AgentLoop]] = None,
        component_factories: Optional[Dict[str, Callable[..., Any]]] = None,
        checkpoint_dir: Optional[str] = None,
        on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
        fast_mode: Optional[bool] = None,
    ):
        self.llm = self._prepare_llm(llm, llms=llms, fallback_llm=fallback_llm, fallback_llms=fallback_llms)
        self.memory = self._prepare_memory(memory)
        self.runtime = runtime if runtime is not None else RestrictedPythonRuntime()
        self.profile = self._prepare_profile(profile)
        self.checkpoint_dir = checkpoint_dir
        self.on_event = on_event
        self.fast_mode = self._is_local_llm(self.llm) if fast_mode is None else bool(fast_mode)
        self.last_run_metrics: Optional[Dict[str, Any]] = None

        # Support passing a list/iterable of tools/functions, a single BaseTool/function, or a ToolRegistry
        if isinstance(tools, (list, tuple, set)):
            registry = (
                ToolRegistry.load_default(runtime=self.runtime)
                if include_default_tools
                else ToolRegistry(runtime=self.runtime)
            )
            for t in tools:
                registered = registry.register(t)
                self._allow_tool_in_profile(registered.name)
            if custom_tools:
                for t in custom_tools:
                    registered = registry.register(t)
                    self._allow_tool_in_profile(registered.name)
            self.tools = registry
        elif tools is None:
            self.tools = ToolRegistry.load_default(runtime=self.runtime)
            if custom_tools:
                for t in custom_tools:
                    registered = self.tools.register(t)
                    self._allow_tool_in_profile(registered.name)
        elif (
            isinstance(tools, BaseTool)
            or inspect.isfunction(tools)
            or (inspect.isclass(tools) and issubclass(tools, BaseTool))
        ):
            registry = (
                ToolRegistry.load_default(runtime=self.runtime)
                if include_default_tools
                else ToolRegistry(runtime=self.runtime)
            )
            registered = registry.register(tools)
            self._allow_tool_in_profile(registered.name)
            if custom_tools:
                for t in custom_tools:
                    reg_custom = registry.register(t)
                    self._allow_tool_in_profile(reg_custom.name)
            self.tools = registry
        else:
            self.tools = tools
            if isinstance(self.tools, ToolRegistry):
                self.tools.set_runtime(self.runtime)
                if custom_tools:
                    for t in custom_tools:
                        registered = self.tools.register(t)
                        self._allow_tool_in_profile(registered.name)

        self._factories = component_factories or {}
        self._dynamic_components: Dict[str, Any] = {}
        self.loop_cls = loop_cls or AgentLoop

        # Setup framework with explicit override -> factory -> default precedence
        self.tool_manager = tool_manager or self._build_component("tool_manager")
        if self.tool_manager is None:
            self.tool_manager = ToolManager(
                self.tools,
                profile=self.profile,
                runtime=self.runtime,
            )

        self.thinker = thinker or self._build_component("thinker")
        if self.thinker is None:
            self.thinker = Thinker(self.llm, profile=self.profile, tool_manager=self.tool_manager)

        self.planner = planner or self._build_component("planner")
        if self.planner is None:
            self.planner = Planner(self.thinker, self.tools, profile=self.profile)

        self.reflector = reflector or self._build_component("reflector")
        if self.reflector is None:
            self.reflector = Reflector(self.thinker, profile=self.profile)

        self.actor = actor or self._build_component("actor")
        if self.actor is None:
            self.actor = Actor(
                self.tool_manager,
                thinker=self.thinker,
                runtime=self.runtime,
                reflector=self.reflector,
            )

        self.analyzer = analyzer or self._build_component("analyzer")
        if self.analyzer is None:
            self.analyzer = Analyzer(self.llm, profile=self.profile)

        # Build user-defined custom components (e.g. generator) from factories.
        self._build_dynamic_components()

        self.loop = loop or self._build_component("loop")
        if self.loop is None:
            self.loop = self._construct_loop()

    @staticmethod
    def _is_local_llm(llm_obj: Any) -> bool:
        if isinstance(llm_obj, LocalLLM):
            return True
        if isinstance(llm_obj, LLMRouter):
            return bool(llm_obj.providers) and all(isinstance(p, LocalLLM) for p in llm_obj.providers)
        return False

    def _prepare_llm(
        self,
        llm: Any,
        llms: Optional[Iterable[Any]] = None,
        fallback_llm: Optional[Any] = None,
        fallback_llms: Optional[Iterable[Any]] = None,
    ) -> Any:
        candidates: List[Any] = []
        if llm is not None:
            candidates.append(llm)
        if llms is not None:
            candidates.extend(list(llms))
        if fallback_llm is not None:
            candidates.append(fallback_llm)
        if fallback_llms is not None:
            candidates.extend(list(fallback_llms))

        if not candidates:
            return OpenAILLM()
        if len(candidates) == 1:
            return OpenAILLM(**candidates[0]) if isinstance(candidates[0], dict) else candidates[0]
        return LLMRouter(candidates)

    def _prepare_memory(self, memory):
        if memory is None:
            return MemoryManager()

        required = ("add_interaction", "get_full_context")
        has_required = all(hasattr(memory, name) for name in required)

        if has_required:
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
        runtime, loop_cls, profile, plus any custom components built from other factories.
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
            "runtime": self.runtime,
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
            "memory_manager": self.memory if isinstance(self.memory, MemoryManager) else getattr(self.memory, "backing_memory", None),
            "checkpoint_dir": self.checkpoint_dir,
            "fast_reflect": self.fast_mode,
        }
        loop_kwargs.update(self._dynamic_components)
        return self._call_with_supported_kwargs(self.loop_cls, loop_kwargs)

    def _allow_tool_in_profile(self, tool_name: str) -> None:
        if self.profile and getattr(self.profile, "tool_access", None):
            if "*" not in self.profile.tool_access and tool_name not in self.profile.tool_access:
                self.profile.tool_access.append(tool_name)

    def register_tool(
        self,
        tool_or_func: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        allow_in_profile: bool = True,
        **kwargs: Any,
    ) -> BaseTool:
        """
        Register a new custom tool (BaseTool instance/subclass, @tool function, or plain Python function)
        into the agent's ToolRegistry and ensure the agent can execute it.
        """
        registered = self.tools.register(tool_or_func, name=name, description=description, **kwargs)
        if isinstance(registered, BaseTool) and getattr(registered, "runtime", None) is None:
            registered.set_runtime(self.runtime)
        if allow_in_profile and hasattr(registered, "name"):
            self._allow_tool_in_profile(registered.name)
        return registered

    def register_tools(self, tools: Iterable[Any], allow_in_profile: bool = True) -> List[BaseTool]:
        """Register multiple custom tools or functions at once."""
        return [self.register_tool(t, allow_in_profile=allow_in_profile) for t in tools]

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool by name from the agent's ToolRegistry."""
        if hasattr(self.tools, "unregister"):
            return self.tools.unregister(name)
        return False

    def get_tools_info(self) -> List[Dict[str, Any]]:
        """Return parameter schemas and descriptions for all tools available to this agent."""
        if self.tool_manager and hasattr(self.tool_manager, "get_available_tools_info"):
            return self.tool_manager.get_available_tools_info()
        if hasattr(self.tools, "get_all_tools_info"):
            return self.tools.get_all_tools_info()
        return []

    def tool(
        self,
        name_or_func: Optional[Union[str, Callable]] = None,
        description: Optional[str] = None,
        *,
        name: Optional[str] = None,
        allow_in_profile: bool = True,
        **kwargs: Any,
    ):
        """
        Decorator to define and immediately register a custom tool on this Agent instance.

        Usage:
            @agent.tool
            def my_tool(x: int) -> int:
                '''Doubles x.'''
                return x * 2

            @agent.tool("my_tool", "Doubles x")
            def my_tool(x: int) -> int:
                return x * 2
        """
        if callable(name_or_func):
            t_obj = create_tool(name_or_func, name=name, description=description, **kwargs)
            return self.register_tool(t_obj, allow_in_profile=allow_in_profile)

        resolved_name = name or (name_or_func if isinstance(name_or_func, str) else None)

        def decorator(func: Callable) -> BaseTool:
            t_obj = create_tool(func, name=resolved_name, description=description, **kwargs)
            return self.register_tool(t_obj, allow_in_profile=allow_in_profile)

        return decorator

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

    def set_cognition_pipeline(
        self,
        bootstrap_pipeline_path: Optional[str] = None,
        loop_pipeline_path: Optional[str] = None,
    ):
        """
        Replace default cognition JSON pipelines with custom specs.
        """
        if not hasattr(self.loop, "set_pipeline_specs"):
            raise ValueError("Current loop does not support pipeline specs.")
        self.loop.set_pipeline_specs(
            bootstrap_pipeline_path=bootstrap_pipeline_path,
            loop_pipeline_path=loop_pipeline_path,
        )

    def _build_auto_classification_prompt(self, prompt: str) -> str:
        tools_summary = ""
        tools_info = self.get_tools_info()
        if tools_info:
            lines = [f"- {t.get('name')}: {t.get('description', '')}" for t in tools_info if isinstance(t, dict)]
            if lines:
                tools_summary = "\nAvailable Tools:\n" + "\n".join(lines) + "\n"
        return (
            f"Analyze the following user prompt and decide if it requires using any of the Available Tools or multi-step planning "
            f"(like calling a custom tool, writing files, searching the web, or complex logic), "
            f"or if it is a simple question that can be answered immediately without tools.\n"
            f"{tools_summary}\n"
            f"Prompt: {prompt}\n\n"
            f"Respond with exactly one word: 'PLAN' or 'FAST'."
        )

    def get_last_run_metrics(self) -> Optional[Dict[str, Any]]:
        """Return token usage, latency, retries, failovers, and tool stats for the last run."""
        return dict(self.last_run_metrics) if self.last_run_metrics is not None else None

    async def run(self, prompt: str, session_id: str = None, max_iterations: int = 15, mode: str = "auto") -> str:
        """
        Run the agent with the given prompt.
        mode can be "auto", "plan", "react", or "fast_ans".
        """
        if hasattr(self.llm, "client") and self.llm.client is None:
            if hasattr(self.llm, "init"):
                await self.llm.init()

        if session_id is None:
            session_id = str(uuid.uuid4())

        tracer.begin_run(session_id=session_id, on_event=self.on_event)
        run_status = "success"
        try:
            await self.memory.add_interaction(session_id, "user", prompt)

            actual_mode = mode
            if mode == "auto":
                if self.fast_mode:
                    # Zero-LLM fast heuristic for Local LLM / ERP speed:
                    # Route directly to ReAct (1-2 calls total) when tools exist, avoiding extra classifier call.
                    actual_mode = "react" if self.get_tools_info() else "fast_ans"
                else:
                    classification_prompt = self._build_auto_classification_prompt(prompt)
                    classification = await self.llm.generate(classification_prompt, session_id=None, max_tokens=10)
                    actual_mode = "plan" if "PLAN" in str(classification).upper() else "fast_ans"

            if actual_mode == "fast_ans":
                context = await self.memory.get_full_context(session_id, query=prompt)
                fast_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAnswer directly:"
                fast_answer = await self.llm.generate(fast_prompt, session_id=None)
                await self.memory.add_interaction(session_id, "assistant", fast_answer)
                return fast_answer
            if actual_mode == "react" and hasattr(self.loop, "run_react"):
                return await self.loop.run_react(prompt, self.memory, session_id, max_iterations=max_iterations)
            return await self.loop.run(prompt, self.memory, session_id, max_iterations=max_iterations)
        except Exception:
            run_status = "error"
            raise
        finally:
            self.last_run_metrics = tracer.end_run(status=run_status)

    async def resume(self, session_id: str, max_iterations: int = 15) -> str:
        """Resume an interrupted or checkpointed agent run from `checkpoint_dir`."""
        if not hasattr(self.loop, "resume"):
            raise ValueError("Current loop does not support resume.")
        if hasattr(self.llm, "client") and self.llm.client is None:
            if hasattr(self.llm, "init"):
                await self.llm.init()

        tracer.begin_run(session_id=session_id, on_event=self.on_event)
        run_status = "success"
        try:
            return await self.loop.resume(self.memory, session_id, max_iterations=max_iterations)
        except Exception:
            run_status = "error"
            raise
        finally:
            self.last_run_metrics = tracer.end_run(status=run_status)

    async def run_stream(self, prompt: str, session_id: str = None, max_iterations: int = 15, mode: str = "auto"):
        """
        Streaming variant of run(). Uses loop.run_stream when available.
        """
        if hasattr(self.llm, "client") and self.llm.client is None:
            if hasattr(self.llm, "init"):
                await self.llm.init()

        if session_id is None:
            session_id = str(uuid.uuid4())

        tracer.begin_run(session_id=session_id, on_event=self.on_event)
        run_status = "success"
        try:
            await self.memory.add_interaction(session_id, "user", prompt)

            actual_mode = mode
            if mode == "auto":
                if self.fast_mode:
                    actual_mode = "react" if self.get_tools_info() else "fast_ans"
                else:
                    classification_prompt = self._build_auto_classification_prompt(prompt)
                    classification = await self.llm.generate(classification_prompt, session_id=None, max_tokens=10)
                    actual_mode = "plan" if "PLAN" in str(classification).upper() else "fast_ans"

            if actual_mode == "fast_ans":
                context = await self.memory.get_full_context(session_id, query=prompt)
                fast_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAnswer directly:"
                fast_answer = await self.llm.generate(fast_prompt, session_id=None)
                await self.memory.add_interaction(session_id, "assistant", fast_answer)
                yield {"type": "status", "content": "Fast answer mode"}
                for line in fast_answer.splitlines() or [fast_answer]:
                    yield {"type": "chunk", "content": f"{line}\n"}
                return

            if actual_mode == "react" and hasattr(self.loop, "run_react_stream"):
                async for event in self.loop.run_react_stream(prompt, self.memory, session_id, max_iterations=max_iterations):
                    yield event
                return

            if hasattr(self.loop, "run_stream"):
                async for event in self.loop.run_stream(prompt, self.memory, session_id, max_iterations=max_iterations):
                    yield event
                return

            final = await self.loop.run(prompt, self.memory, session_id, max_iterations=max_iterations)
            yield {"type": "status", "content": "Preparing final response..."}
            for line in final.splitlines() or [final]:
                yield {"type": "chunk", "content": f"{line}\n"}
        except Exception:
            run_status = "error"
            raise
        finally:
            self.last_run_metrics = tracer.end_run(status=run_status)


