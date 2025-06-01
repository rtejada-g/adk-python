# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Union

import graphviz

from ..agents import BaseAgent
from ..agents.llm_agent import LlmAgent
from ..tools.agent_tool import AgentTool
from ..tools.base_tool import BaseTool
from ..tools.function_tool import FunctionTool

logger = logging.getLogger('google_adk.' + __name__)

try:
  from ..tools.retrieval.base_retrieval_tool import BaseRetrievalTool
except ModuleNotFoundError:
  retrieval_tool_module_loaded = False
else:
  retrieval_tool_module_loaded = True


async def build_graph(graph, agent: BaseAgent, highlight_pairs):
  # Gradient with more contrast: Very Dark Blue/Slate to a Lighter Blue. Google Blue for edges.
  gradient_start_color = "#1B2336"    # Very Dark Slate/Blue-Charcoal
  gradient_end_color = "#133874"      # Lighter, Brighter Blue (Material Blue 400)
  highlight_border_color = '#133874'   # Match very dark start of gradient
  highlight_font_color = '#FFFFFF'     # White font
  highlight_edge_color = '#4285F4'    # Standard Google Blue for edges

  light_gray = '#cccccc' # For non-highlighted font

  def get_node_name(tool_or_agent: Union[BaseAgent, BaseTool]):
    if isinstance(tool_or_agent, BaseAgent):
      return tool_or_agent.name
    elif isinstance(tool_or_agent, BaseTool):
      return tool_or_agent.name
    else:
      raise ValueError(f'Unsupported tool type: {tool_or_agent}')

  def get_node_caption(tool_or_agent: Union[BaseAgent, BaseTool]):

    if isinstance(tool_or_agent, BaseAgent):
      return '🤖 ' + tool_or_agent.name
    elif retrieval_tool_module_loaded and isinstance(
        tool_or_agent, BaseRetrievalTool
    ):
      return '🔎 ' + tool_or_agent.name
    elif isinstance(tool_or_agent, FunctionTool):
      tool_name = tool_or_agent.displayName if hasattr(tool_or_agent, 'displayName') and tool_or_agent.displayName else tool_or_agent.name
      return '🔧 ' + tool_name
    elif isinstance(tool_or_agent, AgentTool):
      # AgentTool might also benefit from a displayName if it wraps another agent
      agent_name = tool_or_agent.displayName if hasattr(tool_or_agent, 'displayName') and tool_or_agent.displayName else tool_or_agent.name
      return '🤖 ' + agent_name
    elif isinstance(tool_or_agent, BaseTool): # Catch-all for other BaseTool derived classes
      tool_name = tool_or_agent.displayName if hasattr(tool_or_agent, 'displayName') and tool_or_agent.displayName else tool_or_agent.name
      return '🔧 ' + tool_name
    else:
      logger.warning(
          'Unsupported tool, type: %s, obj: %s',
          type(tool_or_agent),
          tool_or_agent,
      )
      return f'❓ Unsupported tool type: {type(tool_or_agent)}'

  def get_node_shape(tool_or_agent: Union[BaseAgent, BaseTool]):
    if isinstance(tool_or_agent, BaseAgent):
      return 'ellipse'
    elif retrieval_tool_module_loaded and isinstance(
        tool_or_agent, BaseRetrievalTool
    ):
      return 'cylinder'
    elif isinstance(tool_or_agent, FunctionTool):
      return 'box'
    elif isinstance(tool_or_agent, BaseTool):
      return 'box'
    else:
      logger.warning(
          'Unsupported tool, type: %s, obj: %s',
          type(tool_or_agent),
          tool_or_agent,
      )
      return 'cylinder'

  def draw_node(tool_or_agent: Union[BaseAgent, BaseTool]):
    name = get_node_name(tool_or_agent)
    shape = get_node_shape(tool_or_agent)
    caption = get_node_caption(tool_or_agent)
    if highlight_pairs:
      for highlight_tuple in highlight_pairs:
        if name in highlight_tuple:
          graph.node(
              name,
              caption,
              style='filled,rounded', # All highlighted nodes are rounded
              fillcolor=f'{gradient_start_color}:{gradient_end_color}',
              color=highlight_border_color,
              shape=shape,
              fontcolor=highlight_font_color,
          )
          return
    # if not in highlight, draw non-highlighted node
    # It will inherit 'filled,rounded' style from default node_attr
    graph.node(
        name,
        caption,
        shape=shape, 
        # style will be inherited from node_attr
    )

  def draw_edge(from_name, to_name):
    if highlight_pairs:
      for highlight_from, highlight_to in highlight_pairs:
        if from_name == highlight_from and to_name == highlight_to:
          graph.edge(from_name, to_name, color=highlight_edge_color, penwidth="2.0")
          return
        elif from_name == highlight_to and to_name == highlight_from:
          graph.edge(from_name, to_name, color=highlight_edge_color, penwidth="2.0", dir='back')
          return
    # if no need to highlight, color gray
    # Color will be inherited from graph.edge_attr. Using 'normal' arrowhead.
    graph.edge(from_name, to_name, arrowhead='normal')

  #logger.info(f"Building graph for agent: {agent.name} (type: {type(agent)})")
  draw_node(agent)
  for sub_agent in agent.sub_agents:
    #logger.info(f"Processing sub-agent: {sub_agent.name} of {agent.name}")
    await build_graph(graph, sub_agent, highlight_pairs) # Added await here as build_graph is async
    #logger.info(f"Drawing edge from {agent.name} to sub-agent {sub_agent.name}")
    draw_edge(agent.name, sub_agent.name)
  if isinstance(agent, LlmAgent):
    #logger.info(f"Processing tools for LlmAgent: {agent.name}")
    tools = await agent.canonical_tools()
    #if not tools:
      #logger.info(f"No canonical tools found for LlmAgent: {agent.name}")
    for tool in tools:
      tool_name = get_node_name(tool)
      #logger.info(f"Drawing node for tool: {tool_name} (for agent {agent.name})")
      draw_node(tool)
      #logger.info(f"Drawing edge from {agent.name} to tool {tool_name}")
      draw_edge(agent.name, tool_name)
  #else:
    #logger.info(f"Agent {agent.name} is not an LlmAgent, skipping direct tool processing for it.")


async def get_agent_graph(root_agent, highlights_pairs, image=False):
  #logger.info(f"--- get_agent_graph called for root: {root_agent.name} ---")
  #logger.info(f"Highlight pairs received: {highlights_pairs}")
  graph = graphviz.Digraph(
      graph_attr={
          'rankdir': 'LR',
          'bgcolor': '#333537',
          'splines': 'spline',  # Changed from 'curved'
          'concentrate': 'true',
          'overlap': 'false', # Added to prevent node overlap
      },
      node_attr={
          'fontname': 'Arial',
          'fontsize': '12',
          'style': 'filled,rounded', # Default to rounded corners
          'shape': 'box', # Default shape
          'fillcolor': '#424242', # Slightly darker gray for non-highlighted nodes
          'fontcolor': '#E8EAED',
          'color': '#5F6368', # Border color
      },
      edge_attr={
          'color': '#757575', # Darker gray for non-highlighted edges
          'arrowsize': '0.7',
      },
  )
  await build_graph(graph, root_agent, highlights_pairs)
  if image:
    return graph.pipe(format='png')
  else:
    return graph
