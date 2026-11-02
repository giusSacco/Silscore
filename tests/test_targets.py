"""Interface invariants for target-window prediction."""
import csv
import sys

import numpy as np
import pytest

from silscore import SiRNARecord, Silscore, normalize_target, predict_targets, target_records
from silscore.cli import main
from silscore.model import load_default_model
from silscore.prediction import read_prediction_input, read_records, read_target_csv


class GuideScoreModel:
    """Deterministic stand-in exposing the guides received by predict_targets."""

    def predict(self, records):
        return np.asarray([sum(ord(base) for base in record.guide) / 1000 for record in records])


def test_asymmetric_reverse_complement_and_scores_match_explicit_guides():
    target = 'AUGCCGAUACGUGCAUAGCU'
    model = load_default_model()
    rows = predict_targets(model, [('target', target)])
    explicit = [
        SiRNARecord('explicit', '0', 'GCUAUGCACGUAUCGGCAU'),
        SiRNARecord('explicit', '1', 'AGCUAUGCACGUAUCGGCA'),
    ]
    expected = model.predict(explicit)
    assert [row['guide'] for row in rows] == [record.guide for record in explicit]
    np.testing.assert_array_equal([row['score'] for row in rows], expected)
    assert [(row['start'], row['end']) for row in rows] == [(1, 19), (2, 20)]


def test_window_count_and_boundary_for_19_and_20_nt_targets():
    model = GuideScoreModel()
    rows = predict_targets(model, [('nineteen', 'A' * 19), ('twenty', 'C' * 20)])
    assert len(rows) == 3
    assert [(row['target_id'], row['start'], row['end']) for row in rows] == [
        ('nineteen', 1, 19), ('twenty', 1, 19), ('twenty', 2, 20)
    ]


def test_rna_target_normalization_is_silent(capsys):
    sequence = 'ACGUACGUACGUACGUACG'
    assert normalize_target(sequence) == sequence
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''


@pytest.mark.parametrize('input_kind', ['direct', 'csv'])
@pytest.mark.parametrize('sequence', ['ATGCCGATACGTGCATAGCT', ' atgccgauacgugcauagcu '])
def test_cli_warns_when_converting_t_to_rna(tmp_path, monkeypatch, capsys, input_kind, sequence):
    output = tmp_path / 'predictions.csv'
    expected_target = 'AUGCCGAUACGUGCAUAGCU'
    results = []
    for target, expect_warning in [(sequence, True), (expected_target, False)]:
        if input_kind == 'csv':
            input_path = tmp_path / 'targets.csv'
            input_path.write_text(f'id,target\ntarget,{target}\n')
            input_args = ['--input', str(input_path)]
        else:
            input_args = ['--target', target]
        monkeypatch.setattr(sys, 'argv', ['silscore.py', 'predict', *input_args,
                                        '--output', str(output)])
        assert main() == 0
        captured = capsys.readouterr()
        assert captured.out.startswith('Wrote 2 predictions to ')
        if expect_warning:
            assert captured.err.count('Warning:') == 1
            assert 'converted T to U (treated as RNA)' in captured.err
        else:
            assert captured.err == ''
        results.append(output.read_text())
    assert results[0] == results[1]


@pytest.mark.parametrize('sequence', ['A' * 18, 'A' * 18 + 'N', 'A' * 18 + 'X'])
def test_target_validation_rejects_short_or_ambiguous_input(sequence):
    with pytest.raises(ValueError, match='Target'):
        normalize_target(sequence)


@pytest.mark.parametrize('sequence, changes', [
    ('acguacguacguacguacg', ['converted lowercase to uppercase']),
    (' ACGUACGUACGUACGUACG\n', ['removed whitespace']),
    (' acgtacgtacgtacgtacg\n', ['removed whitespace', 'converted lowercase to uppercase',
                               'converted T to U (treated as RNA)']),
])
def test_public_target_api_normalizes_once(sequence, changes, capsys):
    model = load_default_model()
    expected = predict_targets(model, [('sample', 'ACGUACGUACGUACGUACG')])
    actual = predict_targets(model, [('sample', sequence)])
    assert actual == expected
    assert capsys.readouterr().err == f"Warning: target normalized: {', '.join(changes)}.\n"


@pytest.mark.parametrize('sequence', ['A' * 18, 'A' * 18 + 'N'])
def test_public_target_api_rejects_invalid_target_in_batch(sequence):
    with pytest.raises(ValueError, match="Target 'bad'"):
        predict_targets(GuideScoreModel(), [('good', 'A' * 19), ('bad', sequence)])


def test_target_csv_optional_id_and_mixed_csv_rejection(tmp_path):
    target_path = tmp_path / 'targets.csv'
    target_path.write_text('target\n' + 'A' * 19 + '\n')
    assert read_target_csv(target_path) == [('0', 'A' * 19)]

    mixed_path = tmp_path / 'mixed.csv'
    mixed_path.write_text('id,target,guide\nrow,' + 'A' * 19 + ',' + 'A' * 19 + '\n')
    with pytest.raises(ValueError, match='guide'):
        read_target_csv(mixed_path)
    with pytest.raises(ValueError, match='target'):
        read_prediction_input(mixed_path)


def test_legacy_guide_csv_stays_unchanged(tmp_path):
    guide_path = tmp_path / 'guides.csv'
    guide_path.write_text('id,guide\nlegacy,' + 'ACGU' * 4 + 'ACG' + '\n')
    records = read_records(guide_path)
    assert [(record.id, record.guide) for record in records] == [('legacy', 'ACGUACGUACGUACGUACG')]


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('content, message', [
    ('id,guide,efficacy\nx,ACGUACGUACGUACGUACG,0,8\n', 'Malformed CSV row 2'),
    ('id,guide,efficacy\nx,ACGUACGUACGUACGUACG\n', 'Malformed CSV row 2'),
    ('id,guide,guide,efficacy\nx,' + 'A' * 19 + ',' + 'C' * 19 + ',0.8\n', 'Duplicate'),
    ('id,guide,efficacy\n', 'at least one row'),
])
def test_guide_csv_rejects_malformed_input(tmp_path, training, content, message):
    path = tmp_path / 'guides.csv'
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_records(path, training=training)


def test_training_csv_checks_structure_before_split_filter(tmp_path):
    path = tmp_path / 'training.csv'
    path.write_text('guide,efficacy,split\n' + 'A' * 19 + ',0.8,train\n'
                    + 'C' * 19 + ',0.2,test,extra\n')
    with pytest.raises(ValueError, match='Malformed CSV row 3'):
        read_records(path, training=True)


def test_target_records_preserve_target_input_and_coordinate_order():
    records, rows = target_records([('first', 'A' * 20), ('second', 'C' * 19)])
    assert [record.id for record in records] == ['first:1', 'first:2', 'second:1']
    assert [(row['target_id'], row['start']) for row in rows] == [
        ('first', 1), ('first', 2), ('second', 1)
    ]


@pytest.mark.parametrize('content', [
    'target,target\n' + 'A' * 19 + ',' + 'C' * 19 + '\n',
    'target\n' + 'A' * 19 + ',extra\n',
    'id,target\nmissing\n',
])
def test_rejects_ambiguous_or_malformed_target_csv(tmp_path, content):
    path = tmp_path / 'targets.csv'
    path.write_text(content)
    with pytest.raises(ValueError):
        read_prediction_input(path)


def test_cli_long_target_uses_default_model_without_training(tmp_path, monkeypatch):
    output = tmp_path / 'predictions.csv'
    monkeypatch.setattr(sys, 'argv', ['silscore.py', 'predict', '--target', 'AU' * 250,
                                    '--output', str(output)])
    def unexpected_fit(*args, **kwargs):
        pytest.fail('Prediction must load the pretrained model without fitting.')
    monkeypatch.setattr(Silscore, 'fit', unexpected_fit)
    assert main() == 0
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 482
    assert (rows[0]['start'], rows[-1]['end']) == ('1', '500')


@pytest.mark.parametrize('target', ['ACGU', 'A' * 18 + 'N'])
def test_invalid_target_preserves_existing_output(tmp_path, monkeypatch, target):
    output = tmp_path / 'predictions.csv'
    output.write_text('existing result\n')
    monkeypatch.setattr(sys, 'argv', ['silscore.py', 'predict', '--target', target,
                                    '--output', str(output)])
    with pytest.raises(ValueError):
        main()
    assert output.read_text() == 'existing result\n'
