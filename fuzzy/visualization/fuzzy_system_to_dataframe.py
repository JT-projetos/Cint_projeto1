import pandas as pd
from simpful import FuzzySystem, Triangular_MF


def fuzzy_system_to_dataframe(FS: FuzzySystem) -> pd.DataFrame:
    vis = {
        'x': [],
        'y': [],
        'term': [],
        'linguistic_var': [],
    }

    for key in FS._lvs.keys():
        for fs in FS._lvs[key]._FSlist:

            # Fuzzy set is Triangular
            if isinstance(fs._funpointer, Triangular_MF):

                vis['x'].append(fs._funpointer._a)
                vis['y'].append(0)

                vis['x'].append(fs._funpointer._b)
                vis['y'].append(1.0)

                vis['x'].append(fs._funpointer._c)
                vis['y'].append(0)

                vis['term'].append(fs._term)
                vis['linguistic_var'].append(key)

                vis['term'].append(fs._term)
                vis['linguistic_var'].append(key)

                vis['term'].append(fs._term)
                vis['linguistic_var'].append(key)

    df = pd.DataFrame(vis)
    return df
