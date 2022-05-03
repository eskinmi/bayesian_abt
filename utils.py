from typing import List


class Evidence:

    def __init__(self,
                 variants: List[str],
                 n: List[int],
                 converted: List[int]
                 ):
        self.variants = variants
        self.n = n
        self.converted = converted
        self.conversions = [
            round(self.n[ix] / self.converted[ix])
            for ix in range(len(self.variants))
        ]
        self._basic_checks()

    def _basic_checks(self):
        if not len(self.variants) == len(self.n) == len(self.converted):
            raise ValueError('input lengths should be equal.')

    def __repr__(self):
        return F"ExperimentSummary({','.join(self.variants)})"


def generate_pseudo_dataset(size: int = 1000, **kwargs):
    conversions = kwargs.get('conversions', [0.38, 0.35])
    variants = kwargs.get('variants', ['x', 'y'])
    n = [size, size]
    converted = [round(size * c) for c in conversions]
    return Evidence(variants, n, converted)