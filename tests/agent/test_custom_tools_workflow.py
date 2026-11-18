import json
import unittest
from typing import Literal
from pydantic import BaseModel, Field

from phoenix.framework.agent import Agent, BaseTool, ToolResult, CustomTool, tool, create_tool
from phoenix.framework.agent.cognition.schema import (
    PlanSchema,
    TaskSchema,
    TaskExecutionSchema,
    ToolCallSchema,
    ReflectionSchema,
)


class OrderLookupSchema(BaseModel):
    order_id: int = Field(..., description="Numeric ID of the order")
    include_items: bool = Field(default=True, description="Whether to include line items")


class SyncRawReturnTool(BaseTool):
    """A custom BaseTool subclass with a synchronous execute() returning a raw dict (no ToolResult wrapper)."""
    name = "shipping_estimator"
    description = "Estimates shipping cost by country code and weight."

    def execute(self, country: Literal["EG", "US", "DE"], weight_kg: float = 1.0) -> dict:
        rates = {"EG": 5.0, "US": 20.0, "DE": 15.0}
        return {"country": country, "cost_usd": rates.get(country, 25.0) * weight_kg}


class TestUserCustomToolsWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_all_tool_authoring_styles_and_schema_extraction(self):
        # 1. @tool without parentheses + docstring parameter parsing
        @tool
        def calculate_tax(amount: float, rate: float = 0.14) -> float:
            """
            Calculates tax on a given amount.

            Args:
                amount: Base monetary amount before tax.
                rate: Tax percentage as a decimal.
            """
            return round(amount * rate, 2)

        schema1 = calculate_tax.get_parameters_schema()
        self.assertEqual(calculate_tax.name, "calculate_tax")
        self.assertIn("Calculates tax", calculate_tax.description)
        self.assertEqual(schema1["properties"]["amount"]["type"], "number")
        self.assertEqual(schema1["properties"]["amount"]["description"], "Base monetary amount before tax.")
        self.assertEqual(schema1["properties"]["rate"]["default"], 0.14)
        self.assertEqual(schema1["required"], ["amount"])
        # Direct python call still works
        self.assertEqual(calculate_tax(100.0, 0.14), 14.0)

        # 2. @tool with args_schema (Pydantic model)
        @tool(name="lookup_order", description="Fetches order details by ID", args_schema=OrderLookupSchema, read_only=True)
        async def lookup_order(order_id: int, include_items: bool = True) -> dict:
            return {"order_id": order_id, "status": "shipped", "items": ["Phoenix SDK"] if include_items else []}

        schema2 = lookup_order.get_parameters_schema()
        self.assertIn("order_id", schema2["properties"])
        self.assertIn("order_id", schema2["required"])
        self.assertTrue(lookup_order.read_only)

        # 3. Sync BaseTool subclass returning raw dict + Literal enum extraction
        ship_tool = SyncRawReturnTool()
        schema3 = ship_tool.get_parameters_schema()
        self.assertEqual(schema3["properties"]["country"]["enum"], ["EG", "US", "DE"])
        res3 = await ship_tool.invoke(country="EG", weight_kg="2.5", hallucinated_extra_arg="ignored")
        self.assertTrue(res3.success)
        self.assertEqual(res3.output, {"country": "EG", "cost_usd": 12.5})

    async def test_agent_handles_raw_functions_decorators_and_custom_tools_end_to_end(self):
        # Raw python function (not decorated!)
        def convert_currency(amount: float, from_curr: str, to_curr: str = "USD") -> str:
            """Converts an amount between currencies."""
            return f"{amount} {from_curr} = {amount * 2.0} {to_curr}"

        class MockStructuredLLM:
            def __init__(self):
                self.prompts = []

            async def generate(self, prompt: str, **kwargs) -> str:
                self.prompts.append(prompt)
                if "Respond with exactly one word: 'PLAN' or 'FAST'." in prompt:
                    return "PLAN"
                return "Converted 50 EUR to 100.0 USD and greeted Phoenix User."

            async def generate_structured(self, prompt: str, schema_cls, **kwargs):
                self.prompts.append(prompt)
                if schema_cls is PlanSchema:
                    return PlanSchema(
                        objective="Convert currency and greet user with custom tools",
                        tasks=[
                            TaskSchema(
                                task_id="t_custom",
                                description="Use convert_currency and custom_greet",
                                task_type="other",
                                priority="high",
                            )
                        ],
                    )
                if schema_cls is TaskExecutionSchema:
                    return TaskExecutionSchema(
                        thought_process="Call both user-defined tools",
                        tools_to_call=[
                            ToolCallSchema(
                                tool_name="convert_currency",
                                arguments={"amount": "50", "from_curr": "EUR", "to_curr": "USD", "extra_ignored": 123},
                            ),
                            ToolCallSchema(
                                tool_name="custom_greet",
                                arguments={"user_name": "Phoenix User"},
                            ),
                        ],
                        is_complete=True,
                    )
                if schema_cls is ReflectionSchema:
                    return ReflectionSchema(
                        is_task_complete=True,
                        rating=10,
                        feedback="Both custom tools executed accurately.",
                        summary="Currency converted and user greeted.",
                    )
                raise ValueError(f"Unexpected schema {schema_cls}")

        llm = MockStructuredLLM()
        # Pass raw function in custom_tools (keeps default tools AND adds convert_currency)
        agent = Agent(llm=llm, custom_tools=[convert_currency, SyncRawReturnTool])

        # Also register a tool directly via @agent.tool decorator
        @agent.tool("custom_greet", "Greets a user with a custom message")
        def greet_fn(user_name: str) -> str:
            return f"Welcome aboard, {user_name}!"

        # Verify default tools + custom tools are all registered
        tool_names = agent.tools.list_tools()
        self.assertIn("file_read", tool_names)
        self.assertIn("convert_currency", tool_names)
        self.assertIn("shipping_estimator", tool_names)
        self.assertIn("custom_greet", tool_names)

        # Run agent in auto mode (auto-classifier sees custom tools -> selects PLAN -> executes custom tools)
        final_answer = await agent.run("Convert 50 EUR to USD and greet Phoenix User", mode="auto")
        self.assertIn("Converted 50 EUR to 100.0 USD", final_answer)

        # Verify Thinker prompt received the custom tools JSON schema
        combined_prompts = "\n".join(llm.prompts)
        self.assertIn("convert_currency", combined_prompts)
        self.assertIn("custom_greet", combined_prompts)

        # Verify MemoryCell recorded the custom tool outputs
        cell = agent.memory.get_cell("t_custom")
        outputs = [t["output"] for t in cell.get_successful_tools()]
        self.assertIn("50.0 EUR = 100.0 USD", outputs)
        self.assertIn("Welcome aboard, Phoenix User!", outputs)


if __name__ == "__main__":
    unittest.main()
