"""Numerical invariants and persistence of the published model."""
import copy
import warnings
import numpy as np
import pytest

from silscore import SiRNARecord, Silscore
from silscore.model import PotencyModel
from silscore.thermo import TurnerNNBackend

@pytest.fixture
def records():
    rng = np.random.default_rng(314159)
    guides = [''.join(g) for g in rng.choice(list('ACGU'), size=(500, 19))]
    return [SiRNARecord('synthetic', str(i), g, 0.2 + 0.12*(g[0]=='U') + float(rng.normal(0, .12)))
            for i, g in enumerate(guides)]

def test_wc_energy_accounting():
    backend = TurnerNNBackend()
    for guide in ['A'*19, 'C'*19, 'G'*19, 'U'*19, 'ACGUACGUACGUACGUACG']:
        terminal = .45 * ((guide[0] in 'AU') + (guide[-1] in 'AU'))
        assert backend.duplex_dg(guide) == pytest.approx(sum(backend.stack_profile(guide)) + 4.09 + terminal)

def test_fit_intercept_constraint_and_screening(records):
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        model = Silscore().fit(records)
    assert (1, 'U') in model.identity_terms
    assert abs(sum(model.report()['wvec'])) < 1e-12
    residuals = np.array([r.efficacy for r in records]) - model.predict(records)
    assert abs(residuals.mean()) < 1e-12
    assert len(model.predict(records[:7])) == 7

def test_save_load_preserves_absolute_predictions(records, tmp_path):
    model = Silscore().fit(records)
    path = tmp_path / 'model.json'
    model.save(path)
    restored = Silscore.load(path)
    np.testing.assert_array_equal(model.predict(records), restored.predict(records))
    assert restored.identity_terms == model.identity_terms

@pytest.mark.parametrize('mutation', [
    'schema', 'intercept', 'shape', 'constraint', 'duplicate', 'penalty',
    'empty-base', 'multi-base',
])
def test_rejects_corrupt_state(records, mutation):
    data = copy.deepcopy(Silscore().fit(records).to_dict())
    if mutation == 'penalty': data['lam'] = 2.0
    if mutation == 'schema': data['schema'] = 99
    if mutation == 'intercept': data['intercept'] = float('nan')
    if mutation == 'shape': data['w'].pop()
    if mutation == 'constraint': data['w'][0] += .01
    if mutation == 'duplicate': data['identity_terms'].append(data['identity_terms'][0])
    if mutation == 'empty-base': data['identity_terms'][0][1] = ''
    if mutation == 'multi-base': data['identity_terms'][0][1] = 'AC'
    with pytest.raises(ValueError): Silscore.from_dict(data)

def test_invalid_fit_does_not_retain_old_predictions(records):
    model = Silscore().fit(records)
    with pytest.raises(ValueError): model.fit([])
    with pytest.raises(RuntimeError): model.predict(records)

def test_rank_deficient_identity_design_warns_and_retains_fit(records):
    model = PotencyModel(identity_terms=[(1, b) for b in 'ACGU'])
    with pytest.warns(RuntimeWarning, match='penalties resolve the dependency'):
        assert model.fit(records) is model
    prediction = model.predict(records)
    assert np.isfinite(prediction).all()
    assert prediction.mean() == pytest.approx(np.mean([r.efficacy for r in records]))
    assert set(model.report()['identity']) == {f'{base}@1' for base in 'ACGU'}


def test_public_screening_can_keep_all_four_bases():
    rng = np.random.default_rng(702)
    effects = {'A': -.4, 'C': -.15, 'G': .15, 'U': .4}
    records = [SiRNARecord('synthetic', str(i), ''.join(g),
                           effects[g[0]] + float(rng.normal(0, .01)))
               for i, g in enumerate(rng.choice(list('ACGU'), size=(800, 19)))]
    with pytest.warns(RuntimeWarning, match='penalties resolve the dependency'):
        model = Silscore().fit(records)
    assert set(model.identity_terms) == {(1, base) for base in 'ACGU'}
    assert np.isfinite(model.predict(records)).all()


def test_nonunique_penalized_fit_warns_and_remains_usable(tmp_path):
    records = [SiRNARecord('synthetic', str(i), 'A' * 19, float(y))
               for i, y in enumerate(np.linspace(.1, .9, 40))]
    with pytest.warns(RuntimeWarning, match='Penalized fit is not unique'):
        model = Silscore().fit(records)
    prediction = model.predict(records)
    np.testing.assert_allclose(prediction, .5, atol=1e-12, rtol=0)
    assert abs(sum(model.report()['wvec'])) < 1e-12
    path = tmp_path / 'nonunique.json'
    model.save(path)
    np.testing.assert_array_equal(Silscore.load(path).predict(records), prediction)

def test_invalid_sequences_and_labels(records):
    model = Silscore().fit(records)
    for guide in ['A'*18, 'T'*19, 'a'*19, 'N'*19]:
        with pytest.raises(ValueError): model.predict([SiRNARecord('synthetic', 'invalid', guide)])
    invalid = [SiRNARecord('synthetic', str(i), r.guide, float('nan')) for i,r in enumerate(records)]
    with pytest.raises(ValueError): Silscore().fit(invalid)


def test_public_model_rejects_unrecorded_variants():
    for options in ({'intercetta': False}, {'backend': object()}, {'lam': 2}):
        with pytest.raises(TypeError):
            Silscore(**options)


def test_explicit_labels_failure_invalidates_fit(records):
    model = Silscore().fit(records)
    with pytest.raises(ValueError):
        model.fit(records, y=np.zeros(len(records)))
    with pytest.raises(RuntimeError):
        model.predict(records)
