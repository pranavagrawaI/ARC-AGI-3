"""Decision counterexamples: no GPU calls or source/card mutations."""
import copy
from policy import assess_screen, exact_greedy, module, BENCH, validation_decision

fixtures = module(BENCH / 'test_assess_screen.py', 'existing_gate_fixtures')


def screen():
    r = copy.deepcopy(fixtures.BASE)
    for ev in r['configs'].values():
        ev['log'].update(prune_load_weights=[("{'dropped': 36864, 'renumbered': 258048, 'routers': 48}", '48', '448')],
                         model_gib=[73.9], mamba_cache_mode=['align'])
        ev['greedy'] = [{'game': f'game{i}', 'attempt': a, 'status': 200, 'prompt_tokens': 10000,
                         'text': f'fixed answer {i}'} for i in range(4) for a in (0, 1)]
    return r


def test_exact_success_allows_screen():
    r = assess_screen(screen())
    assert r['user_plan']['decision']['status'] == 'PASS'
    assert r['user_plan']['decision']['p_plus'] == 'S2'


def test_launched_noise_allowance_cannot_satisfy_user_plan():
    r = screen()
    r['configs']['S2']['greedy'][1]['text'] = 'different cached result'
    r['decisions']['greedy_S2cached_vs_S1'] = 1
    out = assess_screen(r)
    assert out['launched_card']['decision']['status'] == 'PASS'
    assert out['user_plan']['decision']['status'] == 'FAIL'


def test_matching_http_errors_are_not_correct_outputs():
    r = screen()
    for n in ('S1', 'S2', 'S3'):
        r['configs'][n]['greedy'][0].update(status=500, text='same error')
    assert not exact_greedy(r['configs'], ['S1', 'S2'])
    assert assess_screen(r)['user_plan']['decision']['status'] == 'FAIL'


def test_worker_pruning_missing_invalidates_baseline():
    r = screen()
    r['configs']['S0']['log']['prune_load_weights'] = []
    assert assess_screen(r)['user_plan']['decision']['status'] == 'INVALID'


def test_align_missing_disqualifies_prefix():
    r = screen()
    for n in ('S2', 'S3'):
        r['configs'][n]['log']['mamba_cache_mode'] = []
    assert assess_screen(r)['user_plan']['decision']['status'] == 'FAIL'


def validation(levels=(51, 48), throughput=(423, 450)):
    return {f'P+{i}': {'issues': [], 'per_game': {'game': {}}, 'levels': l,
                     'serving': {'busy': {'gen_tok_s': g}}}
            for i, (l, g) in enumerate(zip(levels, throughput), 1)}


def test_validation_requires_both_mechanism_passes_and_pair_levels():
    assert validation_decision(validation())['candidate'] == 'P+'
    assert validation_decision(validation(throughput=(420, 460)))['candidate'] == 'P'
    assert validation_decision(validation(levels=(45, 45)))['candidate'] == 'P'


def test_incomplete_validation_is_invalid_without_crashing():
    r = validation()
    r['P+2'] = {'issues': ['missing benchmark']}
    assert validation_decision(r)['status'] == 'INVALID'


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('PASS', name)
