import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import graph  
from langchain_core.runnables.graph import MermaidDrawMethod

docs_dir = "docs"
if not os.path.exists(docs_dir):
    os.makedirs(docs_dir)

graph_image = graph.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)

with open(os.path.join(docs_dir, "graph.png"), "wb") as f:
    f.write(graph_image)

print("Graph visualization has been saved to docs/graph.png") 