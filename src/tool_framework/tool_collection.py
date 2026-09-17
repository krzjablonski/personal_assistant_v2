from typing import List
from tool_framework.i_tool import ITool


class ToolCollection:
    def __init__(self, tools: List[ITool]):
        """Keep the supplied tool list as the collection available to an agent."""
        self.tools = tools

    def get_tool(self, tool_name: str) -> ITool:
        """Return the first tool with the requested name, or raise ValueError if absent."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool {tool_name} not found")

    def get_tools(self) -> List[ITool]:
        """Return the live list of tools available in this collection."""
        return self.tools
