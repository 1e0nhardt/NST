import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import recorder
import os

# filepath = r'results\gatys\common_1-6-11-18-25_adain\metrics.csv'
# filepath = r'results\gatys\common_1-6-11-18-25_mean\metrics.csv'
# filepath = r'results\gatys\common_1-6-11-18-25_std\metrics.csv'
# filepath = r'results\gatys\common_1-6-11-18-25_cov\metrics.csv'
# filepath = r'results\gatys\common_1-6-11-18-25_calc_tmp_linear2\metrics.csv'
# filepath = r'results\gatys\common_1-6-11-18-25_cov_ir2\metrics.csv'
# filepath = r'results\gatys\pyramid_1-6-11-18-25_cov\metrics.csv'
# filepath = r'results/nnst/common_25-18-11-6-1_remd/metrics.csv'


filepaths = []
# filepaths.append('results/TestMetrics/pama')
# filepaths.append('results/TestMetrics/Test_AdaAttn')
# filepaths.append('results/TestMetrics/Test_ArtFlow-AdaIN')
# filepaths.append('results/TestMetrics/Test_ArtFlow-WCT')
# filepaths.append('results/TestMetrics/Test_StyleTr2')
# filepaths.append('results/TestMetrics/Test_WCT')
# filepaths.append('results/TestMetrics/Test_WCT_0.2')
# filepaths.append('results/TestMetrics/Test_STROTSS')
# filepaths.append(r'results/TestMetrics/nnst/common_25-18-11-6-1_remd')
# filepaths.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_mean')
# filepaths.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_l2')
filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_cov+remd2pyr/')
# filepaths.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_std')
# filepaths.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_adain')
# filepaths.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_cov')
# filepaths.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_efdm')
# filepaths.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_gatysdivc')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov')
# filepaths.append(r'results/TestMetrics/gatys/common_1-6-11-18-25_cov_weights_1155')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_highkeep')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_nn')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_2')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs64')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_cov-1pyr-300')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs64-1pyr-300')
# filepaths.append(r'results/TestMetrics/gatys/pyramid_1-6-11-18-25_cov_fs-split')
# filepaths.append('results/TestMetrics/nnfm/common_25-18-11-6-1_FS_nn')
# filepaths.append('results/TestMetrics/nnfm/common_25-18-11-6-1_FS_remd')
# filepaths.append(r'results/TestMetrics/Test_NNST')



for filepath in filepaths:
    df = pd.read_csv(filepath + '/metrics.csv', header=0)

    print(df.describe())
    mean = df.describe().iloc[1].values

    expr_name = os.path.dirname(filepath + '/metrics.csv')

    # recorder.add_row('Name', 'Optimization', 'SSIM', 'LPIPS', 'L_m', 'L_m before', 'L_std', 'L_std before', 'L_cov', 'L_cov before')
    recorder.add_row(expr_name, *mean)

