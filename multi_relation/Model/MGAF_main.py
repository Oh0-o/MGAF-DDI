import tensorflow as tf
from utility.parser import parse_args
from utility.helper import *
from utility.batch_test import *
from time import time
from MGAF import MGAF
import numpy as np
import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
# 屏蔽Python警告
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*tensorflow.*")
warnings.filterwarnings("ignore", message=".*Converting sparse IndexedSlices to a dense Tensor.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
# 设置TensorFlow日志级别
tf.get_logger().setLevel('ERROR')


def set_scientific_style():
    """设置科研风格的绘图参数"""
    # 检查可用的样式
    available_styles = plt.style.available

    # 尝试不同的科研样式
    preferred_styles = ['seaborn-v0_8-paper', 'seaborn-paper', 'seaborn-whitegrid', 'seaborn', 'ggplot', 'classic']

    for style in preferred_styles:
        if style in available_styles:
            plt.style.use(style)
            print(f"Using style: {style}")
            break
    else:
        print("Using default matplotlib style")

    # 手动设置参数 - 科研论文风格
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'figure.figsize': (7, 5),
        'figure.titlesize': 14,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'errorbar.capsize': 3,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })

    # 设置seaborn风格
    try:
        sns.set_style("whitegrid", {'axes.edgecolor': 'black', 'grid.color': '.8'})
        sns.set_context("paper", font_scale=1.0)
    except:
        pass


def plot_training_history(train_history, save_path='./results/'):
    """绘制训练历史图"""
    try:
        if not train_history['epoch']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = train_history['epoch']

        # 子图1：损失函数
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_history['loss'], 'b-', label='Total Loss', linewidth=1.5)
        ax1.plot(epochs, train_history['base_loss'], 'r--', label='Base Loss', linewidth=1.0)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('(a) Training Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # 子图2：验证准确率
        ax2 = axes[0, 1]
        ax2.plot(epochs, train_history['valid_accuracy'], 'g-', linewidth=1.5, marker='o', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('(b) Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # 子图3：验证AUC
        ax3 = axes[1, 0]
        if 'valid_auc' in train_history and train_history['valid_auc']:
            ax3.plot(epochs, train_history['valid_auc'], 'm-', linewidth=1.5, marker='s', markersize=4, label='AUC')
        if 'valid_aupr' in train_history and train_history['valid_aupr']:
            ax3.plot(epochs, train_history['valid_aupr'], 'c--', linewidth=1.5, marker='^', markersize=4, label='AUPR')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('(c) Validation AUC/AUPR', fontsize=12, fontweight='bold')
        if 'valid_auc' in train_history or 'valid_aupr' in train_history:
            ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # 子图4：各损失分量
        ax4 = axes[1, 1]
        ax4.plot(epochs, train_history['kge_loss'], 'y-', label='KGE Loss', linewidth=1.0)
        ax4.plot(epochs, train_history['reg_loss'], 'k--', label='Reg Loss', linewidth=1.0)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Loss', fontsize=11)
        ax4.set_title('(d) Loss Components', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f'{save_path}/training_history.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{save_path}/training_history.pdf', bbox_inches='tight')
        plt.savefig(f'{save_path}/training_history.eps', format='eps', bbox_inches='tight')
        plt.close()

        print(f"Training history saved to {save_path}/training_history.png")

    except Exception as e:
        print(f"Error plotting training history: {e}")


def build_run_name(args):
    if hasattr(args, 'exp_name') and args.exp_name:
        run_name = str(args.exp_name).strip()
    else:
        lr_text = ('%g' % args.lr).replace('.', 'p')
        run_name = f"bs{args.batch_size}_lr{lr_text}_emb{args.embed_size}"

    run_name = run_name.replace(' ', '_').replace('/', '_')
    if int(getattr(args, 'add_timestamp', 1)) == 1:
        run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return run_name


if __name__ == "__main__":
    args = parse_args()
    train_dict = data_generator.train_dict
    valid_dict = data_generator.valid_dict
    test_dict = data_generator.test_dict
    print('检索train, valid, test数据完毕。')

    tf.compat.v1.set_random_seed(2025)
    np.random.seed(2025)

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"Detected preset CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, ignore --gpu_id={args.gpu_id}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_drugs'] = data_generator.n_drugs
    config['n_relations'] = data_generator.n_relations

    "Load the KG triplets."
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['sparse_adj_list'] = data_generator.sparse_adj_list

    valid_feed = data_generator.valid_feed
    test_feed = data_generator.test_feed

    t0 = time()
    model = MGAF(data_config=config, args=args)

    saver = tf.compat.v1.train.Saver()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    """
    *********************************************************
    Train with validation.
    """
    best_epoch = 0
    best_valid_metrics = None
    best_valid_results = None
    patience = args.patience if hasattr(args, 'patience') else 10
    patience_counter = 0

    # 为每次实验创建独立目录，避免并行实验互相覆盖
    run_name = build_run_name(args)
    results_root = getattr(args, 'results_root', './results')
    results_dir = os.path.join(results_root, args.dataset, run_name)
    os.makedirs(results_dir, exist_ok=True)
    best_ckpt_path = os.path.join(results_dir, f"best_model_{args.dataset}.ckpt")
    print(f"Experiment outputs dir: {results_dir}")
    print(f"Best checkpoint path: {best_ckpt_path}")

    # 设置科学绘图风格
    set_scientific_style()

    # 用于存储训练历史
    train_history = {
        'epoch': [],
        'loss': [],
        'base_loss': [],
        'kge_loss': [],
        'reg_loss': [],
        'valid_accuracy': [],
        'valid_auc': [],
        'valid_aupr': []
    }

    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            btime = time()

            batch_data = data_generator.generate_train_batch()
            feed_dict = data_generator.generate_train_feed_dict(model, batch_data)

            _, batch_loss1, batch_base_loss, batch_kge_loss1, batch_reg_loss1, \
                batch_loss2, batch_kge_loss2, batch_reg_loss2 = model.train(sess, feed_dict=feed_dict)

            loss = loss + batch_loss1 + batch_loss2
            base_loss += batch_base_loss
            kge_loss = batch_kge_loss1 + batch_kge_loss2 + kge_loss
            reg_loss = reg_loss + batch_reg_loss1 + batch_reg_loss2

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
            epoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
        print(perf_str)

        if epoch % args.verbose == 0:
            drugs_to_valid = list(data_generator.valid_dict.keys())
            valid_ret = validate(sess, model, drugs_to_valid, train_dict, valid_dict, valid_feed,
                                 drop_flag=False, batch_test_flag=batch_test_flag)

            drugs_to_test = list(data_generator.test_dict.keys())
            test_ret = test3(sess, model, drugs_to_test, train_dict, test_dict, test_feed,
                             drop_flag=False, batch_test_flag=batch_test_flag)

            print("Validation results:", valid_ret)
            print("Test results:", test_ret)

            current_valid_metric = valid_ret.get('accuracy', 0)

            # 存储训练历史
            train_history['epoch'].append(epoch)
            train_history['loss'].append(float(loss))
            train_history['base_loss'].append(float(base_loss))
            train_history['kge_loss'].append(float(kge_loss))
            train_history['reg_loss'].append(float(reg_loss))
            train_history['valid_accuracy'].append(float(valid_ret.get('accuracy', 0)))
            train_history['valid_auc'].append(float(valid_ret.get('auc', 0)))
            train_history['valid_aupr'].append(float(valid_ret.get('aupr', 0)))

            if best_valid_metrics is None or current_valid_metric > best_valid_metrics:
                best_valid_metrics = current_valid_metric
                best_epoch = epoch
                patience_counter = 0
                best_valid_results = valid_ret.copy()

                save_path = saver.save(sess, best_ckpt_path)
                print(f"Best model saved at epoch {epoch} with validation accuracy {best_valid_metrics:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} with validation accuracy {best_valid_metrics:.4f}")
                    break

    # 绘制训练历史图
    plot_training_history(train_history, results_dir)

    # 加载最佳模型
    print(f"Loading best model from epoch {best_epoch}...")
    saver.restore(sess, best_ckpt_path)

    # 使用增强的可视化测试函数
    drugs_to_test = list(data_generator.test_dict.keys())
    if getattr(args, 'no_viz', False):
        import utility.batch_test as bt
        final_ret = bt.test3(sess, model, drugs_to_test, train_dict, test_dict, test_feed,
                             drop_flag=False, batch_test_flag=batch_test_flag)
        print("[no_viz] skipped visualizations; evaluated via test3.")
    else:
        final_ret = test_with_visualization(sess, model, drugs_to_test, train_dict, test_dict, test_feed,
                                            drop_flag=False, batch_test_flag=batch_test_flag,
                                            save_path=results_dir,
                                            auc_json_path=getattr(args, 'results_json', None))

    print("Final test results with best model:", final_ret)

    # 保存详细结果到文件
    results_dict = {
        'run_info': {
            'run_name': run_name,
            'results_dir': results_dir,
            'gpu_id_arg': int(args.gpu_id),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')
        },
        'best_epoch': best_epoch,
        'best_validation_accuracy': float(best_valid_metrics),
        'best_validation_results': {k: float(v) for k, v in best_valid_results.items()},
        'final_test_results': {k: float(v) for k, v in final_ret.items()},
        'model_parameters': {
            'embed_size': args.embed_size,
            'layer_size': args.layer_size,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'dataset': args.dataset
        },
        'training_history': train_history
    }

    with open(f'{results_dir}/final_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    # CV stability mode: append one metric line (per fold) to the shared jsonl.
    # 每个 fold 保留全部标量指标，5 折合一即可支撑显著性检验（配对t检验/Wilcoxon）。
    if args.results_json:
        rec = {
            'fold': args.output_tag or ('fold_%s' % getattr(args, 'fold', 0)),
            'best_epoch': best_epoch,
            'best_valid_accuracy': float(best_valid_metrics),
        }
        for key, value in final_ret.items():
            try:
                rec[key] = float(value)
            except (TypeError, ValueError):
                pass
        with open(args.results_json, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"CV metric record appended -> {args.results_json}")

    print(f"Results saved to {results_dir}/")