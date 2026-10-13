import argparse
import os


DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'Data')
) + os.sep

def parse_args():
    parser = argparse.ArgumentParser(description="Run MGAF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default=DEFAULT_DATA_PATH,
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for cross-validation (0-4).')

    parser.add_argument('--dataset', nargs='?', default='DDInter',#数据集
                        help='Choose a dataset from {deepddi_data,collected_data,DDInter}')
    parser.add_argument('--verbose', type=int, default=3,
                        help='Interval of evaluation.')
    parser.add_argument('--eval_verbose', type=int, default=0,
                        help='0: disable evaluation detail logs, 1: enable')
    parser.add_argument('--eval_warn', type=int, default=0,
                        help='0: disable AUPR warning logs, 1: enable')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=128,#csv嵌入维度
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[256]',
                        help='Output sizes of every layer')

    parser.add_argument('--dnn_size', nargs='?', default='[257,256,128,3]',#类别
                        help='The size of DNN predictor.')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-6,1e-5,1e-4]',
                        help='Regularization')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='mgaf',
                        help='default mgaf')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='bi',
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=1,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--mess_dropout', nargs='?', default='[0]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--margin', type=float, default=1,
                        help='the score margin between pos and neg samples')
    parser.add_argument('--B', type=int, default=64,
                        help='the number of shared relation matrix')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss focusing parameter (xi).')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss balance parameter (alpha_f).')
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--exp_name', nargs='?', default='(0_fold)',
                        help='Experiment name used in output directory.')
    parser.add_argument('--results_root', nargs='?', default='./results',
                        help='Root directory for experiment outputs.')
    parser.add_argument('--add_timestamp', type=int, default=1,
                        help='1 to append timestamp to run directory, 0 to disable.')

    # CV stability run control (optional). When --data_subdir is set, the loader
    # reads train/valid/test.txt under data_path/dataset/<data_subdir>/ instead of
    # the default fixed train{fold}.txt files.
    parser.add_argument('--data_subdir', type=str, default=None,
                        help='Data subdirectory (e.g. cv10/fold_0) for per-fold runs; None keeps default split.')
    parser.add_argument('--output_tag', type=str, default='',
                        help='Suffix for fold tag written into results_json.')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip heavy per-fold visualizations (T-SNE/ROC/radar).')
    parser.add_argument('--results_json', type=str, default='',
                        help='If set, append a JSON line with final metrics for this fold.')

    return parser.parse_known_args()[0]