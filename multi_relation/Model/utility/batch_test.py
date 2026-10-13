import utility.metrics as metrics
from utility.parser import parse_args
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.manifold import TSNE
from utility.loader_mgaf import MGAF_loader


# 设置科研风格 - 使用更兼容的方式
def set_scientific_style():
    """设置科研风格的绘图参数"""
    # 检查可用的样式
    available_styles = plt.style.available

    # 尝试不同的科研样式
    preferred_styles = ['seaborn-paper', 'seaborn-whitegrid', 'seaborn', 'ggplot', 'classic']

    for style in preferred_styles:
        if style in available_styles:
            plt.style.use(style)
            print(f"Using style: {style}")
            break
    else:
        print("Using default matplotlib style")

    # 手动设置参数
    rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.fancybox': True,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'figure.figsize': (8, 6),
        'figure.titlesize': 16,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'errorbar.capsize': 3
    })

    # 设置seaborn风格（如果可用）
    try:
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.1)
    except:
        pass


# 设置样式
set_scientific_style()

cores = multiprocessing.cpu_count() // 2
args = parse_args()
data_generator = MGAF_loader(args=args, path=args.data_path + args.dataset)
batch_test_flag = False

DEFAULT_EVAL_VERBOSE = bool(getattr(args, 'eval_verbose', 0))
DEFAULT_EVAL_WARN = bool(getattr(args, 'eval_warn', 0))

ITEM_NUM = data_generator.n_drugs
N_TRAIN, N_VALID, N_TEST = data_generator.n_train, data_generator.n_valid, data_generator.n_test
BATCH_SIZE = args.batch_size


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def evaluate_model(sess, model, drugs_to_test, data_feed, drop_flag=False, batch_test_flag=False,
                   return_features=False, verbose=None, warn=None):
    """通用评估函数，可用于验证集和测试集"""
    pool = multiprocessing.Pool(cores)

    batch_size = BATCH_SIZE * 2
    test_drugs = drugs_to_test
    n_test_drugs = len(test_drugs)
    n_drug_batchs = n_test_drugs // batch_size + 1

    all_features = []
    all_labels = []
    all_preds = []
    all_pred_scores = []

    for u_batch_id in range(n_drug_batchs):
        start = u_batch_id * batch_size
        end = (u_batch_id + 1) * batch_size
        drug_batch = test_drugs[start: end]
        item_batch = range(ITEM_NUM)

        feed_dict1 = data_generator.generate_test_feed_dict(model=model,
                                                            drug_batch=drug_batch,
                                                            item_batch=item_batch,
                                                            test_feed=data_feed,
                                                            drop_flag=drop_flag)
        rate_batch1 = model.eval(sess, feed_dict=feed_dict1)

        feed_dict2 = data_generator.generate_test_feed_dict(model=model,
                                                            drug_batch=drug_batch,
                                                            item_batch=item_batch,
                                                            test_feed=data_feed[-1::-1],
                                                            drop_flag=drop_flag)
        rate_batch2 = model.eval(sess, feed_dict=feed_dict2)
        rate_batch = rate_batch1 + rate_batch2

        # 获取特征（如果模型支持）
        if return_features and hasattr(model, 'get_features'):
            features = model.get_features(sess, feed_dict1)
            all_features.append(features)

        # 获取label
        label = []
        for i in range(len(data_feed[0])):
            label.append(data_generator.adj_multi[data_feed[0][i]][data_feed[1][i]] - 1)

        all_labels.extend(label)
        pred_type = np.argmax(rate_batch, axis=1)
        all_preds.extend(pred_type)
        all_pred_scores.append(rate_batch)

    pool.close()

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    if all_pred_scores:
        all_pred_scores = np.vstack(all_pred_scores)

    if return_features and all_features:
        all_features = np.vstack(all_features)

    if verbose is None:
        verbose = DEFAULT_EVAL_VERBOSE
    if warn is None:
        warn = DEFAULT_EVAL_WARN

    result_all, result_eve = evaluate(all_preds, all_pred_scores, all_labels, data_generator.type_num,
                                      verbose=verbose, warn=warn)

    if return_features:
        return result_all, all_features, all_labels, all_preds, all_pred_scores
    return result_all


def validate(sess, model, drugs_to_test, train_dict, valid_dict, valid_feed, drop_flag=False, batch_test_flag=False,
             verbose=None, warn=None):
    """验证集评估函数 - 只返回汇总指标"""
    return evaluate_model(sess, model, drugs_to_test, valid_feed, drop_flag, batch_test_flag,
                          verbose=verbose, warn=warn)


def test3(sess, model, drugs_to_test, train_dict, test_dict, test_feed, drop_flag=False, batch_test_flag=False,
          verbose=None, warn=None):
    """测试集评估函数"""
    return evaluate_model(sess, model, drugs_to_test, test_feed, drop_flag, batch_test_flag,
                          verbose=verbose, warn=warn)


def test_with_visualization(sess, model, drugs_to_test, train_dict, test_dict, test_feed,
                            drop_flag=False, batch_test_flag=False, save_path='./results/',
                            verbose=None, warn=None, auc_json_path=None):
    """测试集评估函数，包含可视化"""
    import os
    os.makedirs(save_path, exist_ok=True)

    # 获取结果和特征
    result_all, features, labels, preds, pred_scores = evaluate_model(
        sess, model, drugs_to_test, test_feed, drop_flag, batch_test_flag, return_features=True,
        verbose=verbose, warn=warn
    )

    # 绘制T-SNE可视化
    plot_tsne_visualization(features, labels, preds, save_path)

    # 绘制ROC曲线
    if pred_scores is not None and len(pred_scores.shape) == 2:
        plot_roc_curves(labels, pred_scores, data_generator.type_num, save_path, auc_json_path=auc_json_path)

    # 绘制雷达图
    _, result_eve = evaluate(preds, pred_scores, labels, data_generator.type_num, verbose=False, warn=warn)
    plot_radar_per_class_metrics(result_eve, save_path=save_path)

    return result_all


def plot_tsne_visualization(features, labels, preds, save_path='./results/'):
    """绘制T-SNE可视化图"""
    import os
    from sklearn.decomposition import PCA
    import matplotlib.colors as mcolors

    if features is None or len(features) == 0:
        print("Warning: No features available for T-SNE visualization")
        return

    if features.shape[1] > 50:  # 如果特征维度太高，先进行PCA
        pca = PCA(n_components=min(50, features.shape[1]))
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features

    # 使用t-SNE降维
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_reduced) - 1), n_iter=1000)
        features_2d = tsne.fit_transform(features_reduced)
    except Exception as e:
        print(f"Error in T-SNE: {e}")
        # 如果TSNE失败，使用PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_reduced)

    # 计算ARI和NMI指标
    try:
        ari_score = adjusted_rand_score(labels, preds)
        nmi_score = normalized_mutual_info_score(labels, preds)
    except Exception as e:
        print(f"Error calculating clustering metrics: {e}")
        ari_score = 0.0
        nmi_score = 0.0

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 子图1：按真实类别着色
    ax1 = axes[0]
    unique_labels = np.unique(labels)
    # colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    cmap = plt.cm.get_cmap("tab20c")
    base_colors = [cmap(i) for i in range(cmap.N)]
    order = np.concatenate([np.arange(0, cmap.N, 2), np.arange(1, cmap.N, 2)])
    base_colors = [base_colors[i] for i in order]
    colors = [base_colors[i % len(base_colors)] for i in range(len(unique_labels))]
    class_color_hex = [mcolors.to_hex(c).lower() for c in colors]

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.sum(mask) > 0:  # 确保有数据点
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=[colors[i]], label=f'Class {label}', alpha=0.7, s=30, edgecolors='w', linewidth=0.5)

    ax1.set_xlabel('Component 1', fontsize=14)
    ax1.set_ylabel('Component 2', fontsize=14)
    ax1.set_title('Visualization by True Labels', fontsize=16, fontweight='bold')
    if len(unique_labels) <= 20:  # 如果类别太多，不显示图例
        ax1.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    # 子图2：按预测正确性着色
    ax2 = axes[1]
    correct_mask = labels == preds
    incorrect_mask = labels != preds

    bright_green = "#00C853"
    bright_red = "#FF1744"
    if bright_green.lower() in class_color_hex:
        bright_green = "#00E676"
    if bright_red.lower() in class_color_hex:
        bright_red = "#FF5252"

    if np.sum(correct_mask) > 0:
        ax2.scatter(features_2d[correct_mask, 0], features_2d[correct_mask, 1],
                    c=bright_green, label='Correct', alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
    if np.sum(incorrect_mask) > 0:
        ax2.scatter(features_2d[incorrect_mask, 0], features_2d[incorrect_mask, 1],
                    c=bright_red, label='Incorrect', alpha=0.7, s=30, edgecolors='w', linewidth=0.5)

    ax2.set_xlabel('Component 1', fontsize=14)
    ax2.set_ylabel('Component 2', fontsize=14)

    # 在标题中添加ARI和NMI指标
    ax2.set_title(f'Visualization by Prediction Accuracy\nARI: {ari_score:.3f}, NMI: {nmi_score:.3f}',
                  fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/tsne_visualization.pdf', bbox_inches='tight')
    plt.close()

    # 保存ARI和NMI指标到文件
    with open(f'{save_path}/clustering_metrics.txt', 'w') as f:
        f.write(f'ARI (Adjusted Rand Index): {ari_score:.4f}\n')
        f.write(f'NMI (Normalized Mutual Information): {nmi_score:.4f}\n')

    print(f"T-SNE visualization saved to {save_path}/tsne_visualization.png")
    print(f"Clustering metrics: ARI={ari_score:.4f}, NMI={nmi_score:.4f}")


def plot_radar_per_class_metrics(result_eve, save_path='./results/'):
    """Plot per-class radar charts for Precision/AUC and Recall/F1."""
    if result_eve is None or len(result_eve) == 0:
        print("Warning: No per-class metrics available for radar plot")
        return

    result_eve = np.asarray(result_eve)
    precision = np.clip(result_eve[:, 4], 0, 1)
    auc = np.clip(np.where(result_eve[:, 2] < 0, 0, result_eve[:, 2]), 0, 1)
    recall = np.clip(result_eve[:, 5], 0, 1)
    f1 = np.clip(result_eve[:, 3], 0, 1)

    n_classes = result_eve.shape[0]
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    precision_closed = np.concatenate([precision, precision[:1]])
    auc_closed = np.concatenate([auc, auc[:1]])
    recall_closed = np.concatenate([recall, recall[:1]])
    f1_closed = np.concatenate([f1, f1[:1]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"polar": True})

    ax1 = axes[0]
    ax1.plot(angles_closed, precision_closed, color="#0F6B4D", linewidth=1.6, alpha=0.85, label="Precision")
    ax1.plot(angles_closed, auc_closed, color="#7B4EA3", linewidth=1.6, alpha=0.85, label="AUC")
    ax1.set_xticks(angles)
    ax1.set_xticklabels([str(i) for i in range(n_classes)], fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], 
                   labels=['0.2', '0.4', '0.6', '0.8', '1.0'],
                   angle=90, fontsize=9, color='gray')
     # ---------- 同心圆刻度尺 ----------
    # 在每个径向刻度圆上，沿 86 个轴方向画向内的小刻度线
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for theta in angles:
            ax1.plot([theta, theta], [r, r - 0.025], 
                     color='gray', alpha=0.3, linewidth=0.4, zorder=0)
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0,
           fontsize=9, frameon=True, fancybox=True, framealpha=0.5)
    ax1.grid(True, alpha=0.2, linewidth=0.5, linestyle='--', color='gray')
    ax1.spines['polar'].set_linewidth(0.6)
    ax1.spines['polar'].set_color('#BBBBBB')

    ax2 = axes[1]
    ax2.plot(angles_closed, recall_closed, color="#2878B5", linewidth=1.6, alpha=0.85, label="Recall")
    ax2.plot(angles_closed, f1_closed, color="#C82423", linewidth=1.6, alpha=0.85, label="F1")
    ax2.set_xticks(angles)
    ax2.set_xticklabels([str(i) for i in range(n_classes)], fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], 
                   labels=['0.2', '0.4', '0.6', '0.8', '1.0'],
                   angle=90, fontsize=9, color='gray')
     # ---------- 同心圆刻度尺 ----------
    # 在每个径向刻度圆上，沿 86 个轴方向画向内的小刻度线
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for theta in angles:
            ax2.plot([theta, theta], [r, r - 0.025], 
                     color='gray', alpha=0.3, linewidth=0.4, zorder=0)
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.00, 1.00), borderaxespad=0.0,
           fontsize=9, frameon=True, fancybox=True, framealpha=0.5)
    ax2.grid(True, alpha=0.2, linewidth=0.5, linestyle='--', color='gray')
    ax2.spines['polar'].set_linewidth(0.6)
    ax2.spines['polar'].set_color('#BBBBBB')

    plt.tight_layout()
    plt.savefig(f"{save_path}/radar_per_class_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Radar metrics plot saved to {save_path}/radar_per_class_metrics.png")

def plot_roc_curves(labels, pred_scores, num_classes, save_path='./results/', auc_json_path=None):
    """绘制ROC曲线。若给出 auc_json_path（5折草率 jsonl），则把宏观AUC（mean±std）标注在图上。"""
    try:
        from sklearn.metrics import roc_curve, auc
        import os
        import json

        # 转换为one-hot编码
        y_one_hot = label_binarize(labels, classes=np.arange(num_classes))

        # 确保pred_scores是二维的
        if len(pred_scores.shape) == 1:
            pred_scores = pred_scores.reshape(-1, 1)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

        for i in range(num_classes):
            # 确保有足够的正样本
            if np.sum(y_one_hot[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_one_hot[:, i], pred_scores[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f'Class {i} (AUC = {roc_auc:.3f})')

        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curves', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # 可选：读取5折稳定性jsonl，把宏观AUC（mean±std）标注到图上
        if auc_json_path and os.path.exists(auc_json_path):
            aucs = []
            with open(auc_json_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if 'auc' in d:
                            aucs.append(float(d['auc']))
                    except Exception:
                        pass
            if aucs:
                macro_mean = float(np.mean(aucs))
                macro_std = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
                ax.text(0.03, 0.03, f'macro-AUC = {macro_mean:.4f} ± {macro_std:.4f}',
                        transform=ax.transAxes, ha='left', va='bottom', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.75, edgecolor='black', boxstyle='round'))
                print(f"[annotation] macro-AUC = {macro_mean:.4f} ± {macro_std:.4f} "
                      f"(mean±std over {len(aucs)} folds, from {auc_json_path})")

        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/roc_curves.pdf', bbox_inches='tight')
        plt.close()
        print(f"ROC curves saved to {save_path}/roc_curves.png")
    except Exception as e:
        print(f"Error plotting ROC curves: {e}")


def roc_aupr_score(y_true, y_score, average="macro", warn=True):
    def _binary_roc_aupr_score(y_true, y_score):
        # 检查是否有正样本
        n_positives = np.sum(y_true)
        if n_positives == 0:
            if warn:
                print(f"⚠️  警告: 该类别没有正样本，AUPR 返回 0.0")
            return 0.0
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        # 检查是否有 NaN 值（由除以 0 导致）
        if np.isnan(precision).any() or np.isnan(recall).any():
            if warn:
                print(f"⚠️  警告: precision_recall_curve 计算出现 NaN 值，可能是预测分数异常")
            return 0.0
        
        # 排序并计算 AUC
        order = np.lexsort((recall, precision))
        precision1, recall1 = precision[order], recall[order]
        
        aupr_score = auc(precision1, recall1)
        
        # 检查最终结果是否有效
        if np.isnan(aupr_score) or np.isinf(aupr_score):
            if warn:
                print(f"⚠️  警告: AUPR 计算结果无效 ({aupr_score})，返回 0.0")
            return 0.0
            
        return aupr_score

    def _average_binary_score(binary_metric, y_true, y_score, average):
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        valid_classes = 0
        
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
            if score[c] > 0:
                valid_classes += 1
        
        # 根据 average 参数选择计算方式
        if average == "micro":
            return np.average(score)
        elif average == "macro":
            return np.mean(score)
        elif average == "weighted":
            # 按每个类别的样本数加权
            weights = np.sum(y_true, axis=0)
            total = np.sum(weights)
            if total > 0:
                return np.average(score, weights=weights)
            else:
                return np.mean(score)
        else:
            return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)



def evaluate(pred_type, pred_score, y_test, event_num, verbose=True, warn=True):
    result_all = {}
    each_eval_type = 6
    result_eve = np.zeros((event_num, 6), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))

    # verbose=True 时打印完整诊断信息
    if verbose:
        print("\n" + "="*60)
        print("📊 数据分布诊断信息")
        print("="*60)
        for i in range(event_num):
            n_samples = np.sum(y_one_hot[:, i])
            print(f"类别 {i}: {n_samples} 个样本 ({n_samples/len(y_test)*100:.2f}%)")
        print("="*60 + "\n")

     #, zero_division=0
    result_all['accuracy'] = accuracy_score(y_test, pred_type)
    result_all['aupr'] = roc_aupr_score(y_one_hot, pred_score, average='micro', warn=warn)
    result_all['auc'] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all['f1'] = f1_score(y_test, pred_type, average='macro', zero_division=0)
    result_all['precision'] = precision_score(y_test, pred_type, average='macro', zero_division=0)
    result_all['recall'] = recall_score(y_test, pred_type, average='macro', zero_division=0)

    # verbose=True 时打印各类别详细指标
    if verbose:
        print("\n" + "="*60)
        print("📈 各类别评估指标")
        print("="*60)
        print(f"{'类别':<6} {'Accuracy':<12} {'AUPR':<12} {'AUC':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
        print("-"*60)

    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_score.take([i], axis=1).ravel(),
                          average=None, warn=warn)
        try:
            result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(),
                                             pred_score.take([i], axis=1).ravel(),
                                             average=None)
        except Exception as ex:
            result_eve[i, 2] = -1
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary', zero_division=0)
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary', zero_division=0)
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary', zero_division=0)

        if verbose:
            print(f"{i:<6} {result_eve[i, 0]:<12.4f} {result_eve[i, 1]:<12.4f} {result_eve[i, 2]:<12.4f} "
                  f"{result_eve[i, 3]:<12.4f} {result_eve[i, 4]:<12.4f} {result_eve[i, 5]:<12.4f}")

    if verbose:
        print("="*60 + "\n")

    return result_all, result_eve
