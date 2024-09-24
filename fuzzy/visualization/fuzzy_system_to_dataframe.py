import numpy as np
import pandas as pd
from simpful import FuzzySystem, LinguisticVariable
from simpful import Triangular_MF, Gaussian_MF


def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


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

                if fs._funpointer._a == fs._funpointer._b:
                    vis['x'].append(fs._funpointer._a)
                    vis['y'].append(1)

                    vis['x'].append(fs._funpointer._c)
                    vis['y'].append(0)

                    for _ in range(2):
                        vis['term'].append(fs._term)
                        vis['linguistic_var'].append(key)

                elif fs._funpointer._b == fs._funpointer._c:
                    vis['x'].append(fs._funpointer._a)
                    vis['y'].append(0)

                    vis['x'].append(fs._funpointer._c)
                    vis['y'].append(1)

                    for _ in range(2):
                        vis['term'].append(fs._term)
                        vis['linguistic_var'].append(key)
                else:
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

            # Fuzzy set is Gaussian
            if isinstance(fs._funpointer, Gaussian_MF):
                xmin, xmax = FS._lvs[key]._universe_of_discourse
                #print(f"{xmax=} | {xmin=}")

                x = np.linspace(xmin, xmax, 50)
                y = gaussian(x, fs._funpointer._mu, fs._funpointer._sigma)

                for x_i, y_i in zip(x, y):
                    vis['x'].append(x_i)
                    vis['y'].append(y_i)

                    vis['term'].append(fs._term)
                    vis['linguistic_var'].append(key)

    df = pd.DataFrame(vis)
    return df
