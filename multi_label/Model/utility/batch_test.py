import utility.metrics as metrics
from utility.parser import parse_args
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings

warnings.filterwarnings('ignore')

from utility.loader_mgaf import MGAF_loader

cores = multiprocessing.cpu_count() // 2

args = parse_args()

data_generator = MGAF_loader(args=args, path=args.data_path + args.dataset)
batch_test_flag = False

ITEM_NUM = data_generator.n_drugs
N_TRAIN, N_VALID, N_TEST = data_generator.n_train, data_generator.n_valid, data_generator.n_test
BATCH_SIZE = args.batch_size
N_LABELS = data_generator.n_labels  # 新增：标签数量


def evaluate_model(sess, model, drugs_to_test, data_feed, drop_flag=False, batch_test_flag=False,
                   return_predictions=False):
    """通用评估函数，使用固定阈值0.5"""
    pool = multiprocessing.Pool(cores)

    batch_size = BATCH_SIZE * 2

    test_drugs = drugs_to_test
    n_test_drugs = len(test_drugs)
    n_drug_batchs = n_test_drugs // batch_size + 1

    all_predictions = []  # 存储所有预测概率
    all_binary_predictions = []  # 存储所有二进制预测标签
    all_labels = []  # 存储所有真实标签

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

        # 使用固定阈值0.5进行评估
        rate_batch1, binary_batch1 = model.eval(sess, feed_dict=feed_dict1)

        feed_dict2 = data_generator.generate_test_feed_dict(model=model,
                                                            drug_batch=drug_batch,
                                                            item_batch=item_batch,
                                                            test_feed=data_feed[-1::-1],
                                                            drop_flag=drop_flag)
        rate_batch2, binary_batch2 = model.eval(sess, feed_dict=feed_dict2)

        # 合并两个方向的预测结果
        rate_batch = (rate_batch1 + rate_batch2) / 2.0  # 取平均概率
        binary_batch = (binary_batch1 + binary_batch2) / 2.0  # 取平均二进制预测

        # 获取真实的多标签
        batch_labels = []
        for i in range(len(data_feed[0])):
            drug1 = data_feed[0][i]
            drug2 = data_feed[1][i]
            # 从adj_multi_labels获取多标签向量
            label_vector = data_generator.adj_multi_labels[drug1][drug2]
            batch_labels.append(label_vector)

        batch_labels = np.array(batch_labels)

        all_predictions.append(rate_batch)
        all_binary_predictions.append(binary_batch)
        all_labels.append(batch_labels)

    pool.close()

    # 合并所有批次的预测和标签
    all_predictions = np.vstack(all_predictions)
    all_binary_predictions = np.vstack(all_binary_predictions)
    all_labels = np.vstack(all_labels)

    # 确保二进制预测是整数（0或1）
    all_binary_predictions = (all_binary_predictions >= 0.5).astype(int)

    # 计算多标签评估指标
    result_all = evaluate_multi_label(all_labels, all_binary_predictions, all_predictions)

    if return_predictions:
        return result_all, all_labels, all_predictions, all_binary_predictions
    else:
        return result_all


def validate(sess, model, drugs_to_test, train_dict, valid_dict, valid_feed, drop_flag=False, batch_test_flag=False):
    """验证集评估函数，使用固定阈值0.5"""
    return evaluate_model(sess, model, drugs_to_test, valid_feed, drop_flag, batch_test_flag)


def test3(sess, model, drugs_to_test, train_dict, test_dict, test_feed, drop_flag=False, batch_test_flag=False):
    """测试集评估函数，使用固定阈值0.5"""
    return evaluate_model(sess, model, drugs_to_test, test_feed, drop_flag, batch_test_flag)


def roc_aupr_score(y_true, y_score, average="macro"):
    """计算AUPR（PR曲线下面积）"""

    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

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
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def evaluate_multi_label(y_true, y_pred_binary, y_pred_score):
    """多标签评估函数"""
    result_all = {}

    # 计算基于样本的指标
    result_all['hamming_loss'] = hamming_loss(y_true, y_pred_binary)
    result_all['jaccard_score'] = jaccard_score(y_true, y_pred_binary, average='samples')
    result_all['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro')
    result_all['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro')
    result_all['precision_micro'] = precision_score(y_true, y_pred_binary, average='micro')
    result_all['precision_macro'] = precision_score(y_true, y_pred_binary, average='macro')
    result_all['recall_micro'] = recall_score(y_true, y_pred_binary, average='micro')
    result_all['recall_macro'] = recall_score(y_true, y_pred_binary, average='macro')
    result_all['accuracy'] = accuracy_score(y_true, y_pred_binary)

    # 计算AUC（需要概率分数）
    try:
        # 对于多标签，使用sklearn的roc_auc_score
        result_all['auc_micro'] = sklearn_roc_auc_score(y_true, y_pred_score, average='micro')
    except Exception as e:
        print(f"Warning: AUC micro calculation failed: {e}")
        result_all['auc_micro'] = -1

    try:
        result_all['auc_macro'] = sklearn_roc_auc_score(y_true, y_pred_score, average='macro')
    except Exception as e:
        print(f"Warning: AUC macro calculation failed: {e}")
        result_all['auc_macro'] = -1

    # 计算AUPR
    try:
        result_all['aupr_micro'] = roc_aupr_score(y_true, y_pred_score, average='micro')
    except Exception as e:
        print(f"Warning: AUPR micro calculation failed: {e}")
        result_all['aupr_micro'] = -1

    try:
        result_all['aupr_macro'] = roc_aupr_score(y_true, y_pred_score, average='macro')
    except Exception as e:
        print(f"Warning: AUPR macro calculation failed: {e}")
        result_all['aupr_macro'] = -1

    # 计算每个标签的指标
    label_metrics = {}
    for label_idx in range(N_LABELS):
        label_true = y_true[:, label_idx]
        label_pred_binary = y_pred_binary[:, label_idx]
        label_pred_score = y_pred_score[:, label_idx]

        if np.sum(label_true) > 0:  # 确保该标签有正样本
            # 计算二分类指标
            label_metrics[label_idx] = {
                'f1': f1_score(label_true, label_pred_binary, zero_division=0),
                'precision': precision_score(label_true, label_pred_binary, zero_division=0),
                'recall': recall_score(label_true, label_pred_binary, zero_division=0),
                'accuracy': accuracy_score(label_true, label_pred_binary),
                'support': np.sum(label_true)
            }

            # 计算AUC和AUPR（需要至少有一个正样本和一个负样本）
            if len(np.unique(label_true)) > 1:
                try:
                    label_metrics[label_idx]['auc'] = sklearn_roc_auc_score(label_true, label_pred_score)
                except:
                    label_metrics[label_idx]['auc'] = -1

                try:
                    label_metrics[label_idx]['aupr'] = roc_aupr_score(label_true, label_pred_score, average=None)
                except:
                    label_metrics[label_idx]['aupr'] = -1
            else:
                label_metrics[label_idx]['auc'] = -1
                label_metrics[label_idx]['aupr'] = -1
        else:
            label_metrics[label_idx] = {
                'f1': 0,
                'precision': 0,
                'recall': 0,
                'accuracy': 0,
                'auc': -1,
                'aupr': -1,
                'support': 0
            }

    result_all['label_metrics'] = label_metrics

    return result_all


def set_scientific_style():
    """设置科研绘图风格（兼容旧版本Matplotlib）"""
    try:
        # 尝试新版本的seaborn样式
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            # 尝试旧版本的seaborn样式
            plt.style.use('seaborn-whitegrid')
        except:
            try:
                # 尝试基本seaborn样式
                plt.style.use('seaborn')
            except:
                # 使用matplotlib默认样式，手动设置参数
                plt.rcParams.update({
                    'figure.facecolor': 'white',
                    'axes.facecolor': 'white',
                    'axes.grid': True,
                    'grid.color': '#E0E0E0',
                    'grid.linewidth': 0.5,
                    'axes.edgecolor': '#404040',
                    'axes.linewidth': 0.8,
                    'font.size': 11,
                    'axes.titlesize': 12,
                    'axes.labelsize': 11,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'legend.fontsize': 10,
                    'figure.dpi': 300,
                    'savefig.dpi': 300,
                    'savefig.format': 'png',
                    'savefig.bbox': 'tight'
                })


def visualize_performance_results(y_true, y_pred_score, y_pred_binary, label_names=None, output_dir='./visualizations'):
    """
    可视化多标签分类性能结果

    Parameters:
    -----------
    y_true : numpy.ndarray
        真实标签，形状为 (n_samples, n_labels)
    y_pred_score : numpy.ndarray
        预测概率分数，形状为 (n_samples, n_labels)
    y_pred_binary : numpy.ndarray
        二进制预测标签，形状为 (n_samples, n_labels)
    label_names : list, optional
        标签名称列表，长度为 n_labels
    output_dir : str
        输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f'Label_{i + 1}' for i in range(n_labels)]

    # 1. 计算每个标签的性能指标
    label_metrics = {}
    for i in range(n_labels):
        y_true_i = y_true[:, i]
        y_pred_binary_i = y_pred_binary[:, i]
        y_pred_score_i = y_pred_score[:, i]

        # 计算各项指标
        label_metrics[i] = {
            'f1': f1_score(y_true_i, y_pred_binary_i, zero_division=0),
            'precision': precision_score(y_true_i, y_pred_binary_i, zero_division=0),
            'recall': recall_score(y_true_i, y_pred_binary_i, zero_division=0),
            'accuracy': accuracy_score(y_true_i, y_pred_binary_i),
            'support': np.sum(y_true_i)
        }

        # 计算AUC和AUPR（需要至少有一个正样本和一个负样本）
        if len(np.unique(y_true_i)) > 1:
            try:
                label_metrics[i]['auc'] = sklearn_roc_auc_score(y_true_i, y_pred_score_i)
            except:
                label_metrics[i]['auc'] = -1

            try:
                precision, recall, _ = precision_recall_curve(y_true_i, y_pred_score_i)
                label_metrics[i]['aupr'] = auc(recall, precision)
            except:
                label_metrics[i]['aupr'] = -1
        else:
            label_metrics[i]['auc'] = -1
            label_metrics[i]['aupr'] = -1

    # 1. 绘制每个标签的F1分数条形图（删除F1=0的标签）
    print("1. Creating F1 score bar chart...")
    fig = create_f1_bar_chart_filtered(label_metrics, label_names, output_dir)

    # 2. 特征降维散点图（T-SNE） - 新增ARI和NMI指标
    print("2. Creating T-SNE visualization with ARI and NMI metrics...")
    fig = create_tsne_visualization_with_metrics(y_true, y_pred_score, y_pred_binary, label_names, output_dir)

    # 3. 绘制雷达图（只显示MACRO-F1和MACRO-RECALL）
    print("3. Creating radar chart with MACRO-F1 and MACRO-RECALL only...")
    fig = create_radar_chart_macro_only(label_metrics, label_names, output_dir)

    # 4. 绘制标签共现热力图
    print("4. Creating label co-occurrence heatmap...")
    fig = create_label_cooccurrence_heatmap(y_true, label_names, output_dir)

    print(f"✓ 所有可视化结果已保存到 {output_dir} 目录")

    # 保存性能指标到文件
    save_performance_metrics(label_metrics, label_names, output_dir)

    return label_metrics


def create_f1_bar_chart_filtered(label_metrics, label_names, output_dir):
    """绘制每个标签的F1分数条形图（删除F1=0的标签）"""
    set_scientific_style()

    # 提取F1分数，过滤掉F1=0的标签
    valid_indices = []
    f1_scores = []
    valid_labels = []

    for i in range(len(label_names)):
        f1 = label_metrics[i]['f1']
        if f1 > 0:  # 只保留F1>0的标签
            valid_indices.append(i)
            f1_scores.append(f1)
            valid_labels.append(label_names[i])

    if len(f1_scores) == 0:
        print("Warning: No labels with F1 > 0")
        return None

    # 排序以便更好地展示
    sorted_indices = np.argsort(f1_scores)
    sorted_f1_scores = np.array(f1_scores)[sorted_indices]
    sorted_labels = [valid_labels[i] for i in sorted_indices]
    sorted_original_indices = [valid_indices[i] for i in sorted_indices]

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))

    # 创建颜色映射
    colors = plt.cm.viridis((sorted_f1_scores - np.min(sorted_f1_scores)) /
                            (np.max(sorted_f1_scores) - np.min(sorted_f1_scores) + 1e-8))

    # 绘制条形图
    bars = ax.barh(range(len(sorted_labels)), sorted_f1_scores,
                   color=colors, edgecolor='black', linewidth=0.5)

    # 添加数值标签
    for i, (bar, score, label_idx) in enumerate(zip(bars, sorted_f1_scores, sorted_original_indices)):
        # 添加F1分数
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', ha='left', fontsize=9)

        # 在左侧添加标签索引
        ax.text(-0.05, bar.get_y() + bar.get_height() / 2,
                f'{label_idx + 1}', va='center', ha='right', fontsize=9, fontweight='bold')

    # 添加平均线
    mean_f1 = np.mean(f1_scores)
    ax.axvline(x=mean_f1, color='red', linestyle='--', linewidth=2,
               label=f'Mean F1: {mean_f1:.3f}')

    # 设置图形属性
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title(f'F1 Score for Each Label (Sorted, {len(f1_scores)}/{len(label_names)} labels)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)

    # 设置x轴范围
    ax.set_xlim([0, min(1.0, max(sorted_f1_scores) * 1.1)])

    # 调整布局
    plt.tight_layout()

    # 保存图形
    plt.savefig(f'{output_dir}/f1_scores_bar_chart_filtered.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/f1_scores_bar_chart_filtered.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ F1分数条形图已保存（过滤后: {len(f1_scores)}/{len(label_names)}个标签）")
    return fig


def get_top_performing_labels(label_metrics, metric='f1', top_k=5):
    """获取性能最好的标签索引"""
    scores = [label_metrics[i][metric] for i in range(len(label_metrics))]
    # 过滤掉负值和零值（表示无法计算或性能很差）
    valid_scores = [(i, score) for i, score in enumerate(scores) if score > 0]
    if len(valid_scores) == 0:
        return []

    valid_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in valid_scores[:top_k]]
    return top_indices


def create_tsne_visualization_with_metrics(y_true, y_pred_score, y_pred_binary, label_names, output_dir,
                                           n_samples=1000):
    """绘制T-SNE降维散点图，显示ARI和NMI指标 - 紧凑布局版本"""
    set_scientific_style()

    # 随机选择样本（如果样本太多）
    if len(y_true) > n_samples:
        indices = np.random.choice(len(y_true), n_samples, replace=False)
        y_true_sampled = y_true[indices]
        y_pred_score_sampled = y_pred_score[indices]
        y_pred_binary_sampled = y_pred_binary[indices]
    else:
        y_true_sampled = y_true
        y_pred_score_sampled = y_pred_score
        y_pred_binary_sampled = y_pred_binary

    # 为每个样本创建一个标签：哪个标签的预测概率最高
    dominant_labels = np.argmax(y_pred_score_sampled, axis=1)

    # 基于真实标签的聚类
    true_clusters = []
    for i in range(len(y_true_sampled)):
        # 将多标签转换为字符串表示作为聚类标签
        true_label_str = ''.join(map(str, y_true_sampled[i].astype(int)))
        true_clusters.append(true_label_str)

    # 基于预测标签的聚类
    pred_clusters = []
    for i in range(len(y_pred_binary_sampled)):
        # 将多标签转换为字符串表示作为聚类标签
        pred_label_str = ''.join(map(str, y_pred_binary_sampled[i].astype(int)))
        pred_clusters.append(pred_label_str)

    # 计算ARI和NMI
    all_clusters = list(set(true_clusters + pred_clusters))
    le = LabelEncoder()
    le.fit(all_clusters)

    true_clusters_encoded = le.transform(true_clusters)
    pred_clusters_encoded = le.transform(pred_clusters)

    # 计算聚类评估指标
    ari_score = adjusted_rand_score(true_clusters_encoded, pred_clusters_encoded)
    nmi_score = normalized_mutual_info_score(true_clusters_encoded, pred_clusters_encoded)

    # 使用所有标签的预测概率作为特征
    features = y_pred_score_sampled

    try:
        # 确保有足够的数据点
        if len(features) <= 1:
            print("Warning: Not enough samples for t-SNE")
            return None

        perplexity = min(30, len(features) - 1)
        if perplexity <= 0:
            print("Warning: Perplexity must be positive")
            return None

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        tsne_results = tsne.fit_transform(features)

        # 统计每个主导标签的数量
        label_counts = np.bincount(dominant_labels, minlength=len(label_names))
        n_labels = len(label_names)

        # 计算聚类统计信息
        unique_true_clusters = len(set(true_clusters))
        unique_pred_clusters = len(set(pred_clusters))
        cluster_ratio = unique_pred_clusters / unique_true_clusters if unique_true_clusters > 0 else 0

        # 创建紧凑布局的图形
        fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(16, 7),
                                                 gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.02})

        # 移除右侧图例子图的坐标轴
        ax_legend.axis('off')

        # 选择颜色
        colors = plt.cm.tab20c(np.linspace(0, 1, n_labels))

        # 绘制散点图 - 在主图上
        legend_handles = []
        legend_labels_list = []

        for i in range(n_labels):
            mask = dominant_labels == i
            if np.sum(mask) > 0:
                # 根据预测置信度调整大小
                confidence_scores = np.max(y_pred_score_sampled[mask], axis=1)
                sizes = 20 + 80 * confidence_scores

                scatter = ax_main.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                                          color=colors[i], s=sizes, alpha=0.7,
                                          edgecolors='white', linewidth=0.5)

                # 为图例创建句柄
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                                 markerfacecolor=colors[i], markersize=8))
                legend_labels_list.append(f'L{i + 1}: {label_names[i]} (n={label_counts[i]})')

        # 设置主图属性
        ax_main.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax_main.set_title('t-SNE Visualization of DDI Samples with Cluster Analysis',
                          fontsize=14, fontweight='bold', pad=15)
        ax_main.grid(True, alpha=0.3, linestyle='--')

        # 设置坐标轴范围，添加5%的边距
        x_min, x_max = tsne_results[:, 0].min(), tsne_results[:, 0].max()
        y_min, y_max = tsne_results[:, 1].min(), tsne_results[:, 1].max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        ax_main.set_xlim(x_min - x_margin, x_max + x_margin)
        ax_main.set_ylim(y_min - y_margin, y_max + y_margin)

        # 在主图内部创建紧凑的颜色条（水平，在底部）
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])

        # 在主图底部添加水平颜色条
        cbar = fig.colorbar(sm, ax=ax_main, orientation='horizontal',
                            fraction=0.035, pad=0.08, shrink=1,aspect=45)
        cbar.set_label('Prediction Confidence', fontsize=10, labelpad=5)
        cbar.ax.tick_params(labelsize=9)

        # 在右侧子图中创建紧凑的图例
        # 根据标签数量调整图例大小和列数
        if n_labels > 20:
            fontsize = 6
            ncol_legend = 2
        elif n_labels > 15:
            fontsize = 7
            ncol_legend = 2
        elif n_labels > 10:
            fontsize = 8
            ncol_legend = 2
        else:
            fontsize = 9
            ncol_legend = 1

        # 计算图例所需高度
        legend_height = min(0.9, n_labels * 0.03 * (2 if ncol_legend == 2 else 1))

        # 在右侧子图中创建图例，贴近主图
        legend = ax_legend.legend(legend_handles, legend_labels_list,
                                  loc='upper left',
                                  fontsize=fontsize,
                                  title='Dominant Labels',
                                  title_fontsize=fontsize + 1,
                                  ncol=ncol_legend,
                                  frameon=True,
                                  framealpha=0.8,
                                  edgecolor='black',
                                  bbox_to_anchor=(0, 1),
                                  handletextpad=0.3,
                                  borderpad=0.5,
                                  labelspacing=0.3)

        # 设置图例背景色
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(0.5)

        # 调整右侧子图的边界以紧贴图例
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)

        # 在主图右上角添加指标信息框
        metrics_text = (
            f'ARI: {ari_score:.3f}\nNMI: {nmi_score:.3f}\n'
            f'True Clusters: {unique_true_clusters}\nPred Clusters: {unique_pred_clusters}'
        )

        ax_main.text(0.98, 0.98, metrics_text,
                     transform=ax_main.transAxes,
                     fontsize=9,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # 调整布局，使图更紧凑
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, wspace=0.01)

        # 保存图形
        plt.savefig(f'{output_dir}/tsne_visualization_compact.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{output_dir}/tsne_visualization_compact.pdf',
                    bbox_inches='tight', facecolor='white')
        plt.close()

        # 打印统计信息
        print(f"✓ T-SNE降维散点图已保存 (紧凑布局)")
        print(f"  布局：左侧主图(75%)，右侧图例(25%)，颜色条在底部")
        print(f"  指标：ARI={ari_score:.3f}, NMI={nmi_score:.3f}")
        print(f"  聚类：真实={unique_true_clusters}, 预测={unique_pred_clusters}")
        print(f"  标签：{n_labels}个，图例字体={fontsize}pt")

        # 返回额外的统计信息
        stats = {
            'ari': ari_score,
            'nmi': nmi_score,
            'true_clusters': unique_true_clusters,
            'pred_clusters': unique_pred_clusters,
            'cluster_ratio': cluster_ratio,
            'n_labels': n_labels,
            'n_samples': len(y_true_sampled)
        }

        return fig, stats
    except Exception as e:
        print(f"Warning: t-SNE visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_radar_chart_macro_only(label_metrics, label_names, output_dir):
    """绘制雷达图，只显示MACRO-F1和MACRO-RECALL两个指标"""
    set_scientific_style()

    # 定义要展示的指标
    metrics_to_show = ['f1', 'recall']
    metric_names = ['MACRO-F1', 'MACRO-RECALL']

    n_metrics = len(metrics_to_show)
    n_labels = len(label_names)

    # 收集数据
    data = np.zeros((n_metrics, n_labels))

    for i in range(n_metrics):
        metric = metrics_to_show[i]
        for j in range(n_labels):
            value = label_metrics[j][metric]
            data[i, j] = max(0, value)  # 处理负值

    # 创建雷达图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='polar')

    # 设置角度 - 每个标签一个角度
    angles = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 设置颜色
    colors = ['#FF6B6B', '#4ECDC4']  # 为两个指标设置不同的颜色

    # 为每个指标绘制一条线
    for i in range(n_metrics):
        # 获取该指标在所有标签上的值
        values = data[i, :].tolist()
        values += values[:1]  # 闭合雷达图

        # 绘制线
        ax.plot(angles, values, 'o-', linewidth=2.5, markersize=6,
                color=colors[i], label=metric_names[i], alpha=0.8)

        # 填充区域
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # 设置刻度标签 - 显示所有标签名称
    ax.set_xticks(angles[:-1])

    # 如果标签太多，调整字体大小
    if n_labels > 20:
        fontsize = 8
    elif n_labels > 10:
        fontsize = 9
    else:
        fontsize = 10

    tick_labels = label_names.copy()

    # 如果标签名称太长，可以截断
    max_label_length = 15
    for i in range(len(tick_labels)):
        if len(tick_labels[i]) > max_label_length:
            tick_labels[i] = tick_labels[i][:max_label_length - 3] + '...'

    ax.set_xticklabels(tick_labels, fontsize=fontsize)

    # 设置网格和标签
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11,
              title='Metrics', title_fontsize=12)

    # 添加标题
    plt.title(f'MACRO-F1 and MACRO-RECALL for {n_labels} DDI Labels',
              fontsize=16, fontweight='bold', pad=25)

    plt.tight_layout()

    # 保存图形
    plt.savefig(f'{output_dir}/radar_chart_macro_only.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/radar_chart_macro_only.pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ 雷达图已保存 (只显示MACRO-F1和MACRO-RECALL，{n_labels}个标签)")
    return fig


def create_label_cooccurrence_heatmap(y_true, label_names, output_dir):
    """绘制标签共现热力图 - 将说明文本放在图表下方"""
    set_scientific_style()

    n_labels = y_true.shape[1]

    # 计算标签共现矩阵
    cooccurrence_matrix = np.zeros((n_labels, n_labels))

    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                # 对角线是每个标签自身的频率
                cooccurrence_matrix[i, j] = np.sum(y_true[:, i])
            else:
                # 计算两个标签同时出现的次数
                cooccurrence_matrix[i, j] = np.sum((y_true[:, i] == 1) & (y_true[:, j] == 1))

    # 创建图形 - 增加高度以容纳下方的说明文字
    fig_height = 12 + max(0, n_labels - 10) * 0.2  # 根据标签数量调整高度
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # 创建热力图
    mask = np.zeros_like(cooccurrence_matrix)
    mask[np.triu_indices_from(mask)] = True  # 只显示下三角部分

    # 使用seaborn绘制热力图
    import seaborn as sns
    heatmap = sns.heatmap(cooccurrence_matrix,
                          mask=mask,
                          cmap='YlOrRd',
                          annot=True,
                          fmt='.0f',
                          square=True,
                          linewidths=0.5,
                          cbar_kws={"shrink": 0.8, "label": "Co-occurrence Count"},
                          ax=ax,
                          annot_kws={"size": 8})  # 减小注释字体大小

    # 设置图形属性
    ax.set_xlabel('Label ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Label ID', fontsize=12, fontweight='bold')
    ax.set_title('Label Co-occurrence Matrix', fontsize=14, fontweight='bold')

    # 设置刻度标签
    tick_labels = [f'L{i + 1}' for i in range(n_labels)]
    ax.set_xticks(np.arange(n_labels) + 0.5)
    ax.set_yticks(np.arange(n_labels) + 0.5)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=9)

    # 添加统计数据说明
    # 计算平均共现次数（不包括对角线）
    cooccurrence_values = cooccurrence_matrix[np.tril_indices(n_labels, k=-1)]
    mean_cooccurrence = np.mean(cooccurrence_values) if len(cooccurrence_values) > 0 else 0
    max_cooccurrence = np.max(cooccurrence_values) if len(cooccurrence_values) > 0 else 0

    # 将对角线说明文本和统计数据放在图表下方
    diag_text = 'Diagonal: Individual label frequency; Lower triangle: Co-occurrence count'
    stats_text = f'Mean co-occurrence: {mean_cooccurrence:.1f}; Max co-occurrence: {max_cooccurrence:.0f}'
    full_text = f'{diag_text} | {stats_text}'

    # 调整热力图位置，为下方文本留出空间
    plt.subplots_adjust(bottom=0.15)  # 底部留出15%的空间

    # 在图表下方添加说明文本
    plt.figtext(0.5, 0.05, full_text,  # 位置：水平居中，垂直方向在底部5%处
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='gray'))

    # 调整布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # 调整rect参数，为底部文本留出空间

    # 保存图形
    plt.savefig(f'{output_dir}/label_cooccurrence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/label_cooccurrence_heatmap.pdf', bbox_inches='tight')
    plt.close()

    # 计算并保存共现统计信息
    cooccurrence_stats = calculate_cooccurrence_stats(cooccurrence_matrix, label_names, output_dir)

    print(f"✓ 标签共现热力图已保存 ({n_labels}个标签)")
    print(f"  平均共现次数: {mean_cooccurrence:.2f}, 最大共现次数: {max_cooccurrence:.0f}")
    print(f"  说明文本已放置在图表下方")
    return fig

def calculate_cooccurrence_stats(cooccurrence_matrix, label_names, output_dir):
    """计算并保存标签共现统计信息"""
    import pandas as pd

    n_labels = len(label_names)

    # 计算每个标签的总共现次数（不包括自身）
    total_cooccurrence = np.sum(cooccurrence_matrix, axis=1) - np.diag(cooccurrence_matrix)

    # 计算最常见的共现对
    cooccurrence_pairs = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            count = cooccurrence_matrix[i, j]
            if count > 0:
                cooccurrence_pairs.append({
                    'Label1': f'L{i + 1}',
                    'Label2': f'L{j + 1}',
                    'Label1_Name': label_names[i],
                    'Label2_Name': label_names[j],
                    'Cooccurrence_Count': count
                })

    # 按共现次数排序
    cooccurrence_pairs.sort(key=lambda x: x['Cooccurrence_Count'], reverse=True)

    # 创建统计DataFrame
    stats_df = pd.DataFrame({
        'Label_ID': [f'L{i + 1}' for i in range(n_labels)],
        'Label_Name': label_names,
        'Individual_Frequency': np.diag(cooccurrence_matrix),
        'Total_Cooccurrence': total_cooccurrence,
        'Cooccurrence_Ratio': total_cooccurrence / np.diag(cooccurrence_matrix)
    })

    # 保存统计信息
    stats_df.to_csv(f'{output_dir}/label_cooccurrence_stats.csv', index=False)

    # 保存前20个最常见的共现对
    top_pairs_df = pd.DataFrame(cooccurrence_pairs[:20])
    top_pairs_df.to_csv(f'{output_dir}/top_cooccurrence_pairs.csv', index=False)

    print(f"✓ 标签共现统计信息已保存")
    return stats_df


def save_performance_metrics(label_metrics, label_names, output_dir):
    """保存性能指标到文件"""
    import pandas as pd

    # 创建数据框
    data = []
    for i, label_name in enumerate(label_names):
        metrics = label_metrics[i]
        data.append({
            'Label': label_name,
            'Label_ID': i + 1,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'AUC': metrics.get('auc', -1),
            'AUPR': metrics.get('aupr', -1),
            'Support': metrics['support']
        })

    df = pd.DataFrame(data)

    # 保存为CSV
    csv_path = f'{output_dir}/performance_metrics.csv'
    df.to_csv(csv_path, index=False)

    # 保存为LaTeX表格
    latex_path = f'{output_dir}/performance_metrics.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))

    print(f"✓ 性能指标已保存到 {csv_path} 和 {latex_path}")


def create_metrics_table(mean_values, metric_names, output_dir):
    """创建指标平均值表格"""
    import pandas as pd

    # 创建数据框
    data = {'Metric': metric_names, 'Average Value': mean_values}
    df = pd.DataFrame(data)

    # 保存为CSV
    csv_path = f'{output_dir}/average_metrics.csv'
    df.to_csv(csv_path, index=False)

    # 保存为LaTeX表格
    latex_path = f'{output_dir}/average_metrics.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))

    print(f"✓ 指标平均值已保存到 {csv_path} 和 {latex_path}")
    return df