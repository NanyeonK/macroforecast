from macrocast import macrocast_start
from macrocast.tree_context import derive_tree_context_from_compiled_spec
from macrocast.specs.compiler import compile_experiment_spec_from_recipe
from macrocast.meta import load_axes_registry


def test_macrocast_start_all_stages() -> None:
    out = macrocast_start()
    assert 'axes' in out
    assert 'registries' in out
    assert 'compile' in out
    assert 'tree_context' in out
    assert 'runs_preview' in out
    assert 'manifest_preview' in out
    assert out['compile']['recipe_id'] == 'minimal_fred_md'
    assert out['tree_context']['compile_path'] == 'recipe_native_experiment_config'
    assert out['manifest_preview']['tree_context']['fixed_axes']


def test_macrocast_start_single_stage() -> None:
    out = macrocast_start(stages=['compile'])
    assert list(out.keys()) == ['selected_stages', 'compile']


def test_derive_tree_context_from_compiled_spec() -> None:
    compiled = compile_experiment_spec_from_recipe('baselines/minimal_fred_md.yaml', preset_id='researcher_explicit')
    ctx = derive_tree_context_from_compiled_spec(compiled.meta_config, load_axes_registry())
    assert 'dataset' in ctx['fixed_axes']
    assert 'horizon' in ctx['sweep_axes']
    assert ctx['fixed_values']['dataset'] == 'fred_md'
    assert ctx['sweep_values']['horizon'] == [1]
