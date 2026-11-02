"""Validation and byte-preserving copies of prepared analysis inputs."""

import csv
import hashlib
import json
from pathlib import Path
import runpy

import pytest


prepare = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/prepare_data.py'))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    raw = tmp_path / 'data/raw'
    raw.mkdir(parents=True)
    for name in ('Mix.csv', 'Taka.csv'):
        (raw / name).write_text('siRNA,mRNA,label,y\n' + 'A' * 19 + ',target,0.8,1\n')
    local = tmp_path / 'data/processed/huesken_canonical_split.csv'
    local.parent.mkdir(parents=True)
    rows = [dict(record_id=key, guide=base * 19, overhang='tt', norm_activity=value,
                 split=split, train_all_human=train, test_all_human=test,
                 train_rodent='0', test_rodent='0', source_note='keep, unchanged')
            for key, base, value, split, train, test in [
                ('a', 'A', '0.800', 'train', '1', '0'),
                ('b', 'C', '0.20', 'test', '0', '1')]]
    incoming = tmp_path / 'incoming.csv'
    write_csv(incoming, rows)
    local.write_bytes(incoming.read_bytes().replace(b',train,', b',test,'))
    return incoming, local


def test_explicit_prepared_csv_is_copied_without_rewriting(tmp_path, inputs):
    incoming, local = inputs
    assert prepare['main'](['--root', str(tmp_path), '--huesken-split', str(incoming)]) == 0
    assert local.read_bytes() == incoming.read_bytes()
    manifest = json.loads((tmp_path / 'data/cache/data_manifest.json').read_text())
    assert manifest['summaries']['huesken_canonical_split'] == {'rows': 2, 'train': 1, 'test': 1}
    assert manifest['inputs'][0]['sha256'] == manifest['outputs'][0]['sha256']


def test_default_input_is_validated_in_place(tmp_path, inputs):
    incoming, local = inputs
    local.write_bytes(incoming.read_bytes())
    before = local.read_bytes()
    assert prepare['main'](['--root', str(tmp_path)]) == 0
    assert local.read_bytes() == before
    manifest = json.loads((tmp_path / 'data/cache/data_manifest.json').read_text())
    assert manifest['outputs'][0]['sha256'] == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize('change, message', [
    ('id_alias', 'missing columns: record_id'),
    ('missing_split', 'split must be train/test'),
    ('flag_alias', 'must be 0/1'),
])
def test_legacy_input_is_rejected_before_writing(tmp_path, inputs, capsys, change, message):
    incoming, local = inputs
    with incoming.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if change == 'id_alias':
        for row in rows:
            row['id'] = row.pop('record_id')
    elif change == 'missing_split':
        rows[0]['split'] = ''
    else:
        rows[0]['train_all_human'] = 'yes'
    write_csv(incoming, rows)
    before = local.read_bytes()
    with pytest.raises(SystemExit) as error:
        prepare['main'](['--root', str(tmp_path), '--huesken-split', str(incoming)])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert local.read_bytes() == before
    assert not (tmp_path / 'data/cache/data_manifest.json').exists()


def test_invalid_evaluation_input_preserves_existing_outputs(tmp_path, inputs):
    incoming, local = inputs
    (tmp_path / 'data/raw/Taka.csv').write_text('siRNA,mRNA,label,y\n' + 'A' * 19 + ',target,nan,1\n')
    manifest = tmp_path / 'data/cache/data_manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('previous manifest\n')
    before = local.read_bytes()
    with pytest.raises(SystemExit):
        prepare['main'](['--root', str(tmp_path), '--huesken-split', str(incoming)])
    assert local.read_bytes() == before
    assert manifest.read_text() == 'previous manifest\n'
