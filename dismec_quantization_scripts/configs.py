"""
configs.py
==========
Dataset configurations for DiSMEC PTQ pipeline.
Update paths for your server before running.
"""

# ============================================================
# SERVER PATHS — update these for your server
# ============================================================

# iitj (10.6.0.45)
IITJ_BASE = '/mnt/sdb/rudra1/dismecpp'

# YV-S03 (10.6.0.59) — Amazon-3M
YVS03_BASE = '/DATA2/rudra1/dismecpp'

# YV-PC-03 (10.6.0.54) — Amazon-3M reverse order
YVPC03_BASE = '/DATA1/rudra1/dismecpp'


# ============================================================
# DATASET CONFIGS
# ============================================================
DATASETS = {
    'eurlex': {
        'base':         IITJ_BASE,
        'weights':      '{base}/eurlex_bow_baseline.model.weights-0-3992',
        'test':         '{base}/data/eurlex/eurlex_test.txt',
        'train':        '{base}/data/eurlex/eurlex_train.txt',
        'model':        '{base}/eurlex_bow_baseline.model',
        'prop_weights': '{base}/python/weights-test-pos.txt',
        'A': 0.55, 'B': 1.5,
        'large':        False,
        'num_features': 5002,    # 5001 features + 1 bias
    },
    'wiki10': {
        'base':         IITJ_BASE,
        'weights':      '{base}/wiki10_bow_baseline.model.weights-0-30937',
        'test':         '{base}/data/wiki10/test.txt',
        'train':        '{base}/data/wiki10/train.txt',
        'model':        '{base}/wiki10_bow_baseline.model',
        'prop_weights': '{base}/python/wiki10-weights-test-pos.txt',
        'A': 0.55, 'B': 1.5,
        'large':        False,
    },
    'amazoncat13k': {
        'base':         IITJ_BASE,
        'weights':      '{base}/amazoncat13k_bow.model.weights-0-13329',
        'test':         '{base}/data/amazoncat13k/test_amazoncat13k.txt',
        'train':        '{base}/data/amazoncat13k/train_amazoncat13k.txt',
        'model':        '{base}/amazoncat13k_bow.model',
        'prop_weights': '{base}/python/amazoncat13k-weights-test-pos.txt',
        'A': 0.55, 'B': 1.5,
        'large':        False,
    },
    'amazon670k': {
        'base':         IITJ_BASE,
        'weights':      '{base}/amazon670k_bow.model.weights-0-670090',
        'test':         '{base}/data/amazon670k/Amazon670K_test.txt',
        'train':        '{base}/data/amazon670k/Amazon670K_train.txt',
        'model':        '{base}/amazon670k_bow.model',
        'prop_weights': '{base}/python/amazon670k-weights-test-pos.txt',
        'A': 0.6, 'B': 2.6,
        'large':        False,
    },
    'delicious200k': {
        'base':         IITJ_BASE,
        'weights':      '{base}/delicious_bow_baseline.model.weights-0-205442',
        'test':         '{base}/data/deliciouslarge/deliciousLarge_test.txt',
        'train':        '{base}/data/deliciouslarge/deliciousLarge_train.txt',
        'model':        '{base}/delicious_bow_baseline.model',
        'prop_weights': '{base}/python/delicious200k-weights-test-pos.txt',
        'A': 0.55, 'B': 1.5,
        'large':        True,
        'chunk_size':   10000,
        'num_features': 782586,
    },
    'amazon3m': {
        'base':         YVS03_BASE,
        'weights_dir':  '/DATA2/rudra1/amazon3m_dismecpp',  # split files
        'test':         '{base}/data/amazon3m/test.txt',
        'train':        '{base}/data/amazon3m/train.txt',
        'model':        '/DATA2/rudra1/amazon3m_dismecpp/amazon3m_combined.model',
        'prop_weights': '{base}/python/amazon3m-weights-test-pos.txt',
        'A': 0.6, 'B': 2.6,
        'large':        True,
        'num_features': 337068,
        'augment_for_bias': True,
    },
}


def get_dataset_config(name: str, base_override: str = None) -> dict:
    """
    Get dataset config with paths resolved.

    Args:
        name:          dataset name
        base_override: override base directory (for different servers)
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Valid: {list(DATASETS.keys())}")

    cfg  = DATASETS[name].copy()
    base = base_override or cfg['base']

    # Resolve {base} placeholders
    for key in ['weights', 'test', 'train', 'model', 'prop_weights']:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = cfg[key].replace('{base}', base)

    cfg['base'] = base
    return cfg
