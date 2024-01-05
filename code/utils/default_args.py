parser_choices = {
    'dataset': ['cifar10'],
    'poison_type': [
        'badnet', 'blend', 'trojan', 'cl', 'ISSBA', 'dynamic', 'tact', 'adp_blend', 'adp_patch', 'wanet', 'sig', 'none'],
    'poison_rate': [i / 1000.0 for i in range(0, 500)],
    'cover_rate': [i / 1000.0 for i in range(0, 500)],
    'cleanser': ['ac', 'ss', 'strip', 'scan', 'ct'],
}

parser_default = {
    'dataset': 'cifar10',
    'poison_type': 'badnet',
    'poison_rate': None,
    'cover_rate': None,
    'alpha': None,
}


seed = 2333
