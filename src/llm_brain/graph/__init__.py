"""Graph database integration for relations."""

from llm_brain.graph.kuzu_graph import KUZU_AVAILABLE

if KUZU_AVAILABLE:
    from llm_brain.graph.kuzu_graph import (
        KuzuGraph as Graph,
    )
    from llm_brain.graph.kuzu_graph import (
        get_graph,
        reset_graph,
    )
else:
    from llm_brain.graph.simple_graph import (
        SimpleGraph as Graph,
    )
    from llm_brain.graph.simple_graph import (
        get_graph,
    )
    from llm_brain.graph.simple_graph import (
        reset_simple_graph as reset_graph,
    )

__all__ = ["Graph", "get_graph", "reset_graph"]
