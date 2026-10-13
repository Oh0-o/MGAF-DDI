import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run MGAF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='ddinter_label',#数据集
                        help='Choose a dataset from {deepddi_data,collected_data,DDInter}')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=128,#csv嵌入维度
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[200]',
                        help='Output sizes of every layer')

    parser.add_argument('--dnn_size', nargs='?', default='[201,128,64,24]',#类别
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

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.2]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--margin', type=int, default=1,
                        help='the score margin between pos and neg samples')
    parser.add_argument('--B', type=float, default=64,
                        help='the number of shared relation matrix')
    parser.add_argument('--patience', type=int, default=30)

    # CV stability run control (optional). When --data_subdir is set, the loader
    # reads train/valid/test.txt under data_path/dataset/<data_subdir>/ instead of
    # the default fixed split folder.
    parser.add_argument('--data_subdir', type=str, default=None,
                        help='Data subdirectory (e.g. cv10/fold_0) for per-fold runs; None keeps default split.')
    parser.add_argument('--output_tag', type=str, default='',
                        help='Suffix for fold-specific checkpoint/result file names.')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip the heavy performance visualizations (used in CV runs).')
    parser.add_argument('--results_json', type=str, default='',
                        help='If set, append a JSON line with macro metrics for this fold.')


    return parser.parse_known_args()[0]