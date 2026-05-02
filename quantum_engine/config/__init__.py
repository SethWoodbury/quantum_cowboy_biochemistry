"""quantum_engine.config — runtime configuration: YAML schema for
``qcb run`` (the declarative pipeline runner).

Cluster / install-time paths (XTB_BIN, MACE_MODELS, PLUMED_KERNEL) live
in :mod:`quantum_engine.site`, not here.
"""
from quantum_engine.config.schema import *  # noqa: F401,F403
