"""Full-graph oracle and policy-audit experiments for the ProMatch decoder.

The modules in this package are diagnostic and experiment infrastructure, not
deployable decoder components.  Import the required submodule directly; this
package deliberately performs no additional eager imports.  This keeps
Matplotlib and the heavier oracle/collection modules out of callers that only
need the parent decoding package.
"""

__all__: list[str] = []
