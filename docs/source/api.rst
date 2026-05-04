API reference
=============

Top-level
---------

.. automodule:: macrocast
   :members: run, replicate, __version__

Execution
---------

.. automodule:: macrocast.core.execution
   :members: execute_recipe, execute_recipe_file, replicate_recipe, ManifestExecutionResult, CellExecutionResult, ReplicationResult

Runtime
-------

.. automodule:: macrocast.core.runtime
   :members: execute_minimal_forecast, materialize_l1, materialize_l2, materialize_l3_minimal, materialize_l4_minimal, materialize_l5_minimal, materialize_l6_runtime, materialize_l7_runtime, materialize_l8_runtime, RuntimeResult

Status vocabulary
-----------------

.. automodule:: macrocast.core.status
   :members:

Custom extensions
-----------------

.. automodule:: macrocast.custom
   :members: register_model, register_preprocessor, register_target_transformer, register_feature_block, register_feature_combiner

Figures
-------

.. automodule:: macrocast.core.figures
   :members:
