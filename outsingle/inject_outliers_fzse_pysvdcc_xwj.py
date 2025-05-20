import math
import os
import sys
import datetime
import random

import numpy as np
import pandas as pd
import scipy

import helpers as h
import optht

now = datetime.datetime.now

nb = np.random.negative_binomial

OUTLIER_TYPES = (
    'b',
    # 'o',
    # 'u',
)

Z_SCORES = (
    # 1,
    2,
    # 3,
    # 4,
    # 5,
    # 6,
    # 7,
    # 8,
)



# standardize = h.standardize
def standardize(data, axis=None):
    return h.transform(data, h._standardize, axis=axis, print_=False, mp=False)


def svd(data, U, s, VT, s_dimension):
    S = scipy.linalg.diagsvd(np.pad(s[:s_dimension], (0, len(s) - s_dimension)), *data.shape)
    return U.dot(S.dot(VT))

def inject(cts, new_cts, true_outlier, outlier_frequency, z_score, outlier_type):
# def inject(fname, outlier_frequency, z_score, outlier_type):
    fname = cts
    fname_new = new_cts

    df = h.csv_to_df(fname, dtype=np.float64)
    sf_ = h.get_size_factors(df).reshape((1, df.values.shape[1]))
    data__ = df.values.astype(np.float64)

    data_sf__ = data__/sf_
    J = data_sf__.shape[0]
    N = data_sf__.shape[1]

    try:
        c4 = np.sqrt(2 / (N - 1)) * math.gamma(N / 2) / math.gamma((N - 1) / 2)
    except OverflowError:  # N too big
        c4 = 1 - 1 / 4 / N - 7 / 32 / (N ** 2) - 19 / 128 / (N ** 3)

    mu__ = np.nanmean(data_sf__, axis=1)
    mu__.shape = (data_sf__.shape[0], 1)

    l_ji__ = np.log2((data_sf__ + 1) / (mu__ + 1))
    l_j_ = np.nanmean(l_ji__, axis=1)
    l_j_.shape = (data_sf__.shape[0], 1)
    l_j_std_ = np.nanstd(l_ji__, axis=1, ddof=1) / c4
    l_j_std_[l_j_std_ == 0] = 0.000000000000001
    l_j_std_.shape = (data_sf__.shape[0], 1)

    zs__ = (l_ji__ - l_j_) / l_j_std_

    # OHT
    U, s, VT = scipy.linalg.svd(zs__)

    s_dimension = optht.optht(zs__, sv=s, sigma=None)
    # Low-rank zs__ representation, i.e. de-noised
    zs_lr_optht__ = svd(zs__, U, s, VT, s_dimension)
    noise = zs__ - zs_lr_optht__

    zs_optht__ = np.empty_like(noise)
    mu_optht_ = np.empty(J)
    std_optht_ = np.empty(J)
    for j in range(J):
        data_j_ = noise[j, :]
        data_j_ = h.clean_zs(data_j_)
        mu = data_j_.mean()
        std = data_j_.std(ddof=1) / c4
        # std = np.sqrt(((data - mu) ** 2).sum() / (data.size - 1))
        if std == 0:
            std = 0.000000000000001

        zs_optht__[j, :] = (data_j_ - mu) / std
        mu_optht_[j] = mu
        std_optht_[j] = std

    print('mu_optht_', mu_optht_.mean())
    print('std_optht_', std_optht_.mean())
    # Create an outlier mask
    data_with_outliers = data__.astype(np.float64)
    outlier_mask = np.zeros_like(data__, dtype=np.int32)
    
    for i in range(N):
        ois = random.sample(range(J), outlier_frequency)
        for oi in ois:
            outlier_value = z_score
            if outlier_type == 'u':
                outlier_value = -outlier_value
            elif outlier_type == 'b':
                outlier_value = random.choice([-1, 1]) * outlier_value
            outlier_mask[oi, i] = outlier_value

    # Reuse zs_optht__ from the original data
    zs_optht_wo__ = np.array(zs_optht__)

    # Generate zs_optht_wo__ from scratch as an ideal SND
    # zs_optht_wo__ = np.random.normal(size=zs_optht__.shape)

    zs_optht_wo__[outlier_mask != 0] = outlier_mask[outlier_mask != 0]

    noise_wo__ = zs_optht_wo__ * std_optht_.reshape((J, 1)) + mu_optht_.reshape((J, 1))
    zs_wo__ = zs_lr_optht__ + noise_wo__

    l_ji_wo__ = zs_wo__ * l_j_std_ + l_j_
    data_sf_wo__ = 2**l_ji_wo__ * (mu__ + 1) - 1

    data_wo__ = data_sf_wo__ * sf_
    data_wo__[data_wo__ < 0] = 0

    outlier_mask_df = pd.DataFrame(
        data=outlier_mask,
        index=df.index,
        columns=df.columns,
    )
    
    df_corrupt = outlier_mask_df.stack().reset_index()
    df_corrupt.columns = ['row', 'col', 'value']
    df_corrupt = df_corrupt.query('value !=0').reset_index(drop=True)
    
    h.save_df_to_csv(df_corrupt, true_outlier)

    # data_with_outliers[outlier_mask != 0] = data_wo__[outlier_mask != 0]
    data_with_outliers = data_wo__
    data_with_outliers[np.isnan(data_with_outliers)] = data__[np.isnan(data_with_outliers)]
    data_with_outliers_df = pd.DataFrame(
        data=np.rint(data_with_outliers).astype(h.COUNT_INT),
        index=df.index,
        columns=df.columns,
    )

    h.save_df_to_csv(data_with_outliers_df, fname_new)


def main():
    print(sys.argv)

    cts=sys.argv[1] #'/media/eys/xwj/soft/outsingle/testdata/df_cts_muscle36.txt'
    new_cts=sys.argv[2]
    true_outlier=sys.argv[3] # osg_simu_df_corrupt_2_001.txt
    z_score=np.int32(sys.argv[4])
    simu=np.int32(sys.argv[5])
    # How many outliers per sample?
    OUTLIER_FREQUENCY = 1
    outlier_type = 'b'
    
    np.random.seed(simu+ z_score*1000)

    # import argparse
    # parser = argparse.ArgumentParser(description='Inject outliers into a dataset.')
    # parser.add_argument('cts', metavar='cts', type=str, nargs=1, help='file with count data')
    # parser.add_argument('new_cts', metavar='new_cts', type=str, nargs=1)
    # parser.add_argument('true_outlier', metavar='true_outlier', type=str, nargs=1)
    # parser.add_argument('simu', metavar='simu', type=str, nargs=1)

    print('==============================================')
    print('Injecting outliers with frequency %d, z-score %f (over-expressed/under-expressed/both?: %s)' % (
    OUTLIER_FREQUENCY, z_score, outlier_type))
    inject(cts, new_cts, true_outlier, OUTLIER_FREQUENCY, z_score, outlier_type)
    print('==============================================')

if __name__ == '__main__':
    main()
