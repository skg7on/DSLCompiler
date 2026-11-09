"""Triton plugin shims for the ``llk_cpu`` entry point.

Upstream discovers out-of-tree backends as ``<value>.compiler`` and
``<value>.driver`` modules.  These modules only re-export the concrete classes
so discovery is a side-effecting-free import: the driver is never active by
default, and no LLK native library is loaded during discovery.
"""
