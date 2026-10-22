"""SPF inference and evidence-backed CMF screening, using only local compact data."""
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent

@lru_cache(maxsize=1)
def package():
    with gzip.open(ROOT / 'bundle/engineering.json.gz', 'rt', encoding='utf-8') as stream:
        return json.load(stream)

def spf_mean(road):
    if not road['available']:
        return None
    b = package()['coefficients']
    eta = (b['intercept'] + b['log_aadf_10000'] * math.log(road['aadf'] / 10000)
           + b['motorway'] * road['motorway'] + b['a_motorway'] * road['a_motorway']
           + b['year_minus_2015'] * (road['baseline_year'] - 2015))
    value = road['length_km'] * math.exp(eta)
    if not math.isfinite(value) or value < 0:
        raise ValueError('SPF inputs could not be evaluated.')
    return value

def metrics(rows):
    n = len(rows)
    return {'n': n, 'predicted': math.fsum(r['predicted'] for r in rows) if n else None,
            'actual': sum(r['actual'] for r in rows) if n else None,
            'mae': math.fsum(abs(r['predicted']-r['actual']) for r in rows)/n if n else None,
            'rmse': math.sqrt(math.fsum((r['predicted']-r['actual'])**2 for r in rows)/n) if n else None}

def checks(ids):
    records = package()['roads']
    rows = [r for key in ids for r in records[key]['history']]
    return {'summary': metrics(rows), 'years': [{'year': year, **metrics([r for r in rows if r['year']==year])}
                                               for year in range(2020, 2026)],
            'protocol': package()['model']['test_protocol'], 'model_id': package()['model_id']}

def road_result(road_id):
    if road_id not in package()['roads']:
        raise ValueError('Road not found in the current network.')
    road = package()['roads'][road_id]
    eligible = road['available'] and not road['motorway'] and not road['a_motorway']
    return {**road, 'id': road_id, 'expected': spf_mean(road), 'cmf_supported': eligible,
            'unavailable_reason': None if road['available'] else 'Complete 2025 traffic coverage is not available for this road unit.',
            'cmf_reason': None if eligible else ('This measure is limited to non-motorway A-roads in this release.'
                                                if road['available'] else 'An eligible SPF baseline is required.'),
            'check_summary': metrics(road['history']), 'cmf': package()['cmf'], 'model_id': package()['model_id']}

def scenario(settings):
    if not isinstance(settings, dict) or set(settings) != {'id', 'measure', 'scenario_type'}:
        raise ValueError('Supply a road, a supported measure and the reference scenario type. Refresh an older workspace page.')
    if not isinstance(settings['id'], str) or not isinstance(settings['measure'], str):
        raise ValueError('Invalid road or measure.')
    if settings['scenario_type'] != 'fixed_reference':
        raise ValueError('Only the fixed-assumption reference scenario is supported.')
    road = road_result(settings['id'])
    if settings['measure'] != package()['cmf']['id']:
        raise ValueError('This measure has no verified factor in the current evidence library.')
    if not road['cmf_supported']:
        return {'status': 'not_supported', 'id': settings['id'], 'measure': settings['measure'],
                'scenario_type': 'fixed_reference',
                'issues': [{'field': 'road', 'kind': 'unsupported', 'message': road['cmf_reason']}]}
    c = package()['cmf']
    baseline = road['expected']
    return {'status': 'estimated', 'id': settings['id'], 'measure': c['id'], 'model_id': package()['model_id'],
            'baseline_year': road['baseline_year'], 'unit': 'injury crashes per year', 'baseline': baseline,
            'cmf': c['factor'], 'after': baseline*c['factor'], 'reduction': baseline*(1-c['factor']),
            'reduction_percent': 100*(1-c['factor']),
            'cmf_95': [c['lower'],c['upper']], 'cmf_only_after_range': [baseline*c['lower'],baseline*c['upper']],
            'range_note': 'CMF-source uncertainty only. Not a local prediction interval or a combined SPF/CMF confidence interval.',
            'scenario_type': 'fixed_reference', 'site_conditions_verified': False,
            'scope': 'Fixed-assumption whole-unit reference estimate at the displayed reference-year traffic. Conditions are assumed, not verified on site. Not a verified local benefit.',
            'conditions_for_estimate': [
                'New cameras cover every section in the selected road unit.',
                'No section has previous speed-camera enforcement.',
                'Speed limits and the road layout do not change at the same time.'],
            'conditions_note': 'Assumed for comparison. Verify on site before implementation.',
            'comparison_assumptions': {'traffic': 'Both scenarios use the same displayed reference-year traffic.',
                'purpose': 'Permanent average-speed enforcement for injury-crash reduction.',
                'assessment': 'Initial screening only. A site assessment is required before implementation.'},
            'source': c['citation'], 'source_url': c['url']}

def meta():
    p = package()
    return {'model_id': p['model_id'], 'baseline_year':p['baseline_year'], 'outcome':p['outcome'],
            'model':p['model'], 'coefficients':p['coefficients'], 'cmf':p['cmf']}
