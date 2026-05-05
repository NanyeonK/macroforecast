Quickstart
==========

Install
-------

.. code-block:: bash

    pip install macroforecast

Run a recipe
------------

.. code-block:: python

    import macroforecast

    result = macroforecast.run("recipe.yaml", output_directory="out/")
    print(result.cells[0].sink_hashes)

Replicate
---------

.. code-block:: python

    replication = macroforecast.replicate("out/manifest.json")
    assert replication.sink_hashes_match

See ``examples/recipes/`` for the full gallery and
``examples/replication/`` for paper-replication scripts.
