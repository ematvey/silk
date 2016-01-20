import pandas as pd

from . import field_normed_by_clicks, Graph, Evaluator


def sm_test():
    df = pd.DataFrame(
            [
                ('ru', 'cosmos', 2, 2),
                ('en', 'cosmos', 4, 2),
                ('ru', 'ararat', 2, 1),
                ('en', 'ararat', 4, 1),
            ],
            columns=['market', 'hotel', 'cost', 'clicks'],
    )

    merge = lambda val, size, pval, ssize: (round((val * size + pval * ssize) / (size + ssize), 5), size + ssize)

    spec = {
        'cost': (field_normed_by_clicks('cost'), merge),
    }

    root = Graph(spec)
    root.child('market', 1).terminal(1)
    root.set_terminal_partitioner(['market', 'hotel'])

    ev = Evaluator(root, df)

    for node in ev.all_nodes:
        if node.key == (('market', 'en'),):
            assert node.value['cost'] == 3.75
        if node.key == (('market', 'ru'),):
            assert node.value['cost'] == 2.25
        if node.key == tuple():
            assert node.value['cost'] == 3.0
