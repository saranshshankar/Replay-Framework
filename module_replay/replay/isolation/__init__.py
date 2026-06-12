"""Pure ModuleSpec -> replay-isolation derivation (RPLY-05).

Each helper turns a ModuleSpec into a string the runner splices into its bash
replay script: the --topics filter, the launch key:=value args, and the
mock-node launch fragment. All helpers are pure (no I/O, no side effects) so
they are trivially testable and the runner stays the only place that touches
the container.
"""
