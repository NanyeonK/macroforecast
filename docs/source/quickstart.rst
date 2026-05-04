Quickstart
==========

Install
-------

.. code-block:: bash

    pip install macrocast

Run a recipe
------------

.. code-block:: python

    import macrocast

    result = macrocast.run("recipe.yaml", output_directory="out/")
    print(result.cells[0].sink_hashes)

Replicate
---------

.. code-block:: python

    replication = macrocast.replicate("out/manifest.json")
    assert replication.sink_hashes_match

See ``examples/recipes/`` for the full gallery and
``examples/replication/`` for paper-replication scripts.
