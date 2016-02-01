from collections import defaultdict

__all__ = ['Graph', 'Evaluator', 'field_normed_by_clicks']


class Slotted(object):
    __slots__ = []

    def __init__(self, *args):
        for k, v in zip(self.__slots__, args):
            setattr(self, k, v)

    def __repr__(self):
        return '{}({})'.format(
                self.__class__.__name__,
                ', '.join(['{}={}'.format(s, getattr(self, s)) for s in self.__slots__])
        )


class KeyMixin(object):
    @property
    def signature(self):
        return tuple(k[0] for k in self.key)

    @property
    def label(self):
        return tuple(k[1] for k in self.key)

    @property
    def key_str(self):
        k = '/'.join('{}={}'.format(*k) for k in self.key)
        if k == '':
            k = 'root'
        return k


class SNode(Slotted, KeyMixin):
    __slots__ = ['key', 'value']


class SParam(SNode):
    __slots__ = ['key', 'value', 'terminal_sm_size']


class SResult(Slotted, KeyMixin):
    __slots__ = ['key', 'metrics', 'data']

    def metrics_annotated(self, evaluator):
        for name, metric in self.metrics.items():
            paths = []
            for p in metric.params:
                key = p.key
                path = [p]
                while len(key) > 0:
                    key = key[:-1]
                    node = evaluator.node_map[key]
                    path.append(node)
                paths.append(path)
            yield name, metric, paths


class SMetric(Slotted):
    __slots__ = ['value', 'native_value', 'native_size', 'params']


class Graph(object):
    def __init__(self, spec=None, parent=None, partitioner=None, parent_sm_size=None, terminal_sm_size=None):
        self.is_child = (parent is not None and
                         parent_sm_size is not None and
                         partitioner is not None)
        self.is_root = (spec is not None)
        assert self.is_child or self.is_root and not (self.is_child and self.is_root)
        self.parent = parent
        self.parent_sm_size = parent_sm_size
        self.terminal_sm_size = None
        self._spec = spec
        self.partitioner = partitioner
        self.terminal_partitioner = None
        self.children = []

    @property
    def key(self):
        return self.partitioner

    @property
    def spec(self):
        if self.parent is not None:
            return self.parent.spec
        return self._spec

    def terminal(self, terminal_sm_size):
        if len(self.children) > 0:
            raise Exception('cannot set terminal on node with children')
        self.terminal_sm_size = terminal_sm_size

    def set_terminal_partitioner(self, partitioner):
        """ This partitioner must be full terminal splitter """
        if not self.is_root:
            raise Exception('cannot set terminal partitioner on non-root node')
        self.terminal_partitioner = partitioner

    @property
    def is_terminal(self):
        return self.terminal_sm_size is not None

    def child(self, partitioner, parent_sm_size):
        if self.is_terminal:
            raise Exception('cannot add child to terminal')
        child = Graph(parent=self, partitioner=partitioner, parent_sm_size=parent_sm_size)
        self.children.append(child)
        return child

    def _eval_partial(self, parent_vals, data):
        if self.is_root:
            grouped_data = [(None, data)]
        else:
            if data[self.partitioner].hasnans:
                raise ValueError('Data has nans at {} column'.format(self.partitioner))
            grouped_data = list(data.groupby(self.partitioner))
        values = defaultdict(lambda: dict())
        for metric, (extractor, merger) in self.spec.items():
            for gv, d in grouped_data:
                value, size = extractor(d)
                if parent_vals is not None:
                    pval = parent_vals[metric]
                    value, total_size = merger(value, size, pval, self.parent_sm_size)
                values[gv][metric] = value

        for gv, d in grouped_data:
            if self.is_root:
                node_coord = None
            else:
                node_coord = (self.key, gv)

            if not self.is_terminal:
                yield False, (node_coord,), values[gv], 0, d

                # children eval passthrough
                for child in self.children:
                    for (
                            isterm,
                            child_coord,
                            child_values,
                            terminal_size,
                            child_data
                    ) in child._eval_partial(values[gv], d):
                        yield isterm, (node_coord, *child_coord), child_values, terminal_size, child_data

            else:
                yield True, (node_coord,), values[gv], self.terminal_sm_size, d

    def eval_graph(self, data):
        if not self.is_root:
            raise Exception('cannot eval on non-root node')

        terminal_nodes = []  # defaultdict(list)
        nodes = []

        for is_terminal, key, values, terminal_sm_size, node_data in self._eval_partial(None, data):
            key = tuple(k for k in key if k is not None)

            nodes.append(SNode(key, values))

            if is_terminal:
                terminal_nodes.append(SParam(key, values, terminal_sm_size))

        return terminal_nodes, nodes

    def _format_self(self):
        return ' '.join([
            (str(self.partitioner) if self.partitioner is not None else 'root'),
            'in:{}'.format(self.parent_sm_size) if self.parent_sm_size is not None else '',
            'out:{}'.format(self.terminal_sm_size) if self.terminal_sm_size is not None else '',
        ])

    def _format_repr(self, level):
        return self._format_self() + ''.join('\n' + '  ' * level + c._format_repr(level + 1) for c in self.children)

    def __repr__(self):
        s = self._format_repr(1)
        if self.terminal_partitioner is not None:
            s += '\n\n terminal: ' + '/'.join(self.terminal_partitioner)
        return s


class Evaluator(object):
    def __init__(self, graph, data, extra_data=None):
        self.graph = graph
        self.data = data
        self.extra_data = extra_data
        self.terminal_nodes, self.all_nodes = self.graph.eval_graph(data)
        self.node_map = {x.key: x for x in self.all_nodes}

        self._ns = defaultdict(dict)
        for n in self.terminal_nodes:
            self._ns[n.signature][n.label] = n

        self._signatures = list(self._ns)
        self._spec_list = list(self.graph.spec.items())

    def __iter__(self):
        tk_seen = set()
        for tk, td in self.data.groupby(self.graph.terminal_partitioner):
            tk_seen.add(tk)
            yield self._one(tk, td)
        if self.extra_data is not None:
            for tk, td in self.extra_data.groupby(self.graph.terminal_partitioner):
                if tk not in tk_seen:
                    yield self._one(tk, td)
                    tk_seen.add(tk)

    @property
    def results(self):
        res = []
        for r in self:
            res.append((r, list(r.metrics_annotated(self))))
        return res

    def _one(self, terminal_key, terminal_data):
        result_metrics = {}
        for metric, (extractor, merger) in self._spec_list:
            sm_params = []

            native_value, native_size = None, None
            value = 0
            size = 0

            _res = extractor(terminal_data)
            if _res is not None:
                native_value, native_size = _res
                value = native_value
                size = native_size

            for mask in self._signatures:
                te = terminal_data[list(mask)].drop_duplicates()

                key = tuple(te.values.tolist()[0])
                # pick first value among duplicates, which is very crude

                param = self._ns[mask][key]
                sm_params.append(param)
                value, size = merger(value, size, param.value[metric], param.terminal_sm_size)

            result_metrics[metric] = SMetric(value, native_value, native_size, sm_params)

        return SResult(
                tuple(zip(self.graph.terminal_partitioner, terminal_key)),
                result_metrics,
                terminal_data,
        )


def field_normed_by_clicks(column_name):
    def extract(d):
        size = d['clicks'].sum()
        if size > 0:
            value = (d[column_name].mul(d['clicks'])).sum() / size
        else:
            value = 0
        return round(value, 4), size

    return extract
