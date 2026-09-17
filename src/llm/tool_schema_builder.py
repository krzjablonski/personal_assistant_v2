"""Shared utility for building JSON Schema tool definitions from ITool instances."""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from tool_framework.i_tool import ToolParameter
    from tool_framework.i_tool import ITool

_JSON_SCHEMA_TYPE_MAP = {
    "float": "number",
    "double": "number",
    "int": "integer",
    "integer": "integer",
    "bool": "boolean",
    "boolean": "boolean",
    "str": "string",
    "string": "string",
    "number": "number",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
    "any": None,
}

_VALID_JSON_SCHEMA_TYPES = {
    "string",
    "number",
    "integer",
    "object",
    "array",
    "boolean",
    "null",
}


def _build_parameter_schema(param: "ToolParameter") -> dict:
    """Describe a tool parameter using JSON Schema types accepted by provider adapters.

    Treat arrays as strings by default and omit unknown or unrestricted types;
    tools with richer shapes should supply an explicit input schema.
    """
    schema = {"description": param.description}
    normalized_type = param.type.lower()
    json_schema_type = _JSON_SCHEMA_TYPE_MAP.get(normalized_type, normalized_type)

    # Unknown internal parameter types should not leak invalid JSON Schema.
    if json_schema_type in _VALID_JSON_SCHEMA_TYPES:
        schema["type"] = json_schema_type
        # Some serving engines' chat templates crash on array schemas that
        # omit `items`, so always emit it. All current list parameters hold
        # strings; tools needing other item types set an explicit input_schema.
        if json_schema_type == "array":
            schema["items"] = {"type": "string"}

    return schema


def build_parameters_schema(tool: "ITool") -> dict:
    """Build a JSON Schema 'parameters'/'input_schema' dict from an ITool.

    If the tool already has `input_schema` set, returns it directly.
    Otherwise constructs the schema from tool.parameters list.
    """
    if tool.input_schema is not None:
        return tool.input_schema

    return {
        "type": "object",
        "properties": {
            param.name: _build_parameter_schema(param)
            for param in tool.parameters
        },
        "required": [
            param.name for param in tool.parameters if param.required
        ],
        "additionalProperties": False,
    }


def tools_to_openai_format(tools: List["ITool"]) -> List[dict]:
    """Convert ITool list to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": build_parameters_schema(tool),
            },
        }
        for tool in tools
    ]


def tools_to_anthropic_format(tools: List["ITool"]) -> List[dict]:
    """Convert ITool list to Anthropic tool format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": build_parameters_schema(tool),
        }
        for tool in tools
    ]
