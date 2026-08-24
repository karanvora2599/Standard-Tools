"""
Standard Tools over the Model Context Protocol.

Run it with the `sqt-mcp` entry point; see `Documentation/18_mcp.md` for
client configuration and `Development/mcp_plan.md` for the measurements the
design follows from.

NOTE ON THE NAME. This package is `standard_quant_tools.mcp` and the SDK it
uses is the top-level `mcp`. Python 3 resolves imports absolutely, so
`import mcp.types` inside this package reaches the SDK and not its own
parent -- the shadowing is a readability hazard rather than a runtime one,
and `tests/mcp/` pins that both resolve correctly.

Only `server` imports the SDK. `catalog`, `schemas`, `resources` and
`prompts` are plain library code, so the tool budget and the resource layer
can be inspected and tested without the SDK installed.

Deliberately no `__all__` and no submodule re-exports: importing `server`
here would pull the SDK in on any `import standard_quant_tools.mcp`, which
is exactly the coupling the split above avoids. Import the submodule you
want.
"""
