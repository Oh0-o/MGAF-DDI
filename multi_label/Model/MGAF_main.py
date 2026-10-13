import tensorflow as tf
from utility.parser import parse_args
from utility.helper import *
from utility.batch_test import *
from time import time
from MGAF import MGAF
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    args = parse_args()
    _tag = ('_' + args.output_tag) if args.output_tag else ''
    _ckpt = f"./best_model_without_triplet_bilinear_{args.dataset}{_tag}.ckpt"
    _res_txt = f"./results_without_triplet_bilinear_{args.dataset}{_tag}.txt"

    train_dict = data_generator.train_dict
    valid_dict = data_generator.valid_dict  # 获取验证集字典
    test_dict = data_generator.test_dict
    print('检索train, valid, test数据完毕。')

    tf.set_random_seed(2025)
    np.random.seed(2025)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_drugs'] = data_generator.n_drugs
    config['n_relations'] = data_generator.n_relations
    config['n_labels'] = data_generator.n_labels  # 新增：标签数量

    "Load the KG triplets."
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['sparse_adj_list'] = data_generator.sparse_adj_list

    valid_feed = data_generator.valid_feed  # 获取验证集数据
    test_feed = data_generator.test_feed

    t0 = time()
    model = MGAF(data_config=config, args=args)  # 使用无三元组和双线性版本

    saver = tf.compat.v1.train.Saver()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    """
    *********************************************************
    Train with validation.
    """
    # 用于早停的变量
    best_epoch = 0
    best_valid_metrics = None
    patience = args.patience if hasattr(args, 'patience') else 10  # 如果未设置，默认为10
    patience_counter = 0

    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        # 训练阶段
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

        # 每个epoch进行验证
        if epoch % args.verbose == 0:
            # 验证集评估
            drugs_to_valid = list(data_generator.valid_dict.keys())

            # 使用固定阈值0.5进行验证
            valid_ret = validate(sess, model, drugs_to_valid, train_dict, valid_dict, valid_feed,
                                 drop_flag=False, batch_test_flag=batch_test_flag)

            # 测试集评估
            drugs_to_test = list(data_generator.test_dict.keys())
            test_ret = test3(sess, model, drugs_to_test, train_dict, test_dict, test_feed,
                             drop_flag=False, batch_test_flag=batch_test_flag)

            print("Validation results:", valid_ret)
            print("Test results:", test_ret)

            # 早停策略
            # 修改：使用多标签评估指标，这里使用micro F1作为早停指标
            current_valid_metric = valid_ret['f1_micro']  # 使用micro F1作为评估指标

            if best_valid_metrics is None or current_valid_metric > best_valid_metrics:
                best_valid_metrics = current_valid_metric
                best_epoch = epoch
                patience_counter = 0

                # 保存最佳模型
                save_path = saver.save(sess, _ckpt)
                print(f"Best model saved at epoch {epoch} with validation micro F1 {best_valid_metrics:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} with validation micro F1 {best_valid_metrics:.4f}")
                    break

    # 加载最佳模型进行最终测试
    print(f"Loading best model from epoch {best_epoch}...")
    saver.restore(sess, _ckpt)

    # 最终测试评估 - 这次返回详细的预测结果用于可视化
    drugs_to_test = list(data_generator.test_dict.keys())
    final_ret, y_true, y_pred_score, y_pred_binary = evaluate_model(
        sess, model, drugs_to_test, test_feed,
        drop_flag=False, batch_test_flag=batch_test_flag,
        return_predictions=True
    )

    print("Final test results with best model:", final_ret)

    # 可选：打印详细的多标签评估指标
    print("\nDetailed Multi-label Evaluation Metrics:")
    print(f"Hamming Loss: {final_ret.get('hamming_loss', 'N/A'):.4f}")
    print(f"Jaccard Score: {final_ret.get('jaccard_score', 'N/A'):.4f}")
    print(f"Micro F1: {final_ret.get('f1_micro', 'N/A'):.4f}")
    print(f"Macro F1: {final_ret.get('f1_macro', 'N/A'):.4f}")
    print(f"Micro AUC: {final_ret.get('auc_micro', 'N/A'):.4f}")
    print(f"Macro AUC: {final_ret.get('auc_macro', 'N/A'):.4f}")
    print(f"Micro AUPR: {final_ret.get('aupr_micro', 'N/A'):.4f}")
    print(f"Macro AUPR: {final_ret.get('aupr_macro', 'N/A'):.4f}")

    # 保存结果到文件
    with open(_res_txt, "w") as f:
        f.write("Ablation Study: Without Triplet Attention and Bilinear Pooling\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Micro F1: {best_valid_metrics:.4f}\n\n")
        f.write("Final Test Results:\n")
        for key, value in final_ret.items():
            if key != 'label_metrics':
                f.write(f"{key}: {value:.4f}\n")

        # 保存每个标签的详细指标
        f.write("\nLabel-wise Metrics:\n")
        for label_idx, metrics in final_ret.get('label_metrics', {}).items():
            f.write(f"Label {label_idx}: ")
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}={metric_value:.4f} ")
            f.write("\n")

    print(f"Results saved to {_res_txt}")

    # 在 CV 模式下，追加一行 Macro 指标到 results_json（含折号、epoch、各 macro 指标）。
    # 每个 fold 仅记录 macro 指标（及整体 accuracy/hamming/jaccard），5 折合一即可支撑显著性检验。
    if args.results_json:
        import json as _json
        rec = {
            'fold': args.output_tag,
            'best_epoch': best_epoch,
        }
        for key, value in final_ret.items():
            if key == 'label_metrics':
                continue
            if 'micro' in key:
                continue
            try:
                rec[key] = float(value)
            except (TypeError, ValueError):
                pass
        with open(args.results_json, 'a') as f:
            f.write(_json.dumps(rec) + "\n")
        print(f"CV metric record appended -> {args.results_json}")

    if args.no_viz:
        print("Skipping visualizations (--no_viz).")
        sys.exit(0)

    # 可视化性能结果
    print("\nGenerating visualizations...")

    # 创建标签名称（根据您的实际标签名称修改）
    label_names = [f'Label_{i + 1}' for i in range(N_LABELS)]

    # 调用可视化函数
    visualize_performance_results(
        y_true,
        y_pred_score,
        y_pred_binary,
        label_names=label_names,
        output_dir=f'./performance_visualizations_{args.dataset}'
    )

    print("All visualizations have been generated successfully!")