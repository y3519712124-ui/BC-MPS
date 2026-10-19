# BC-MPS：分块保序多模态预测集

[English](README.md)

本仓库对应论文 **Block-Conformal Multimodal Prediction Sets for Spatially Separated Hyperspectral–LiDAR Classification**，包含终稿所用的固定30轮实验代码、9份模型权重、配置、运行记录、指标汇总和绘图数据。

## 快速检查

在仓库根目录运行：

```bash
python scripts/verify_artifacts.py
```

这一步只需Python标准库，可检查源文件及权重校验值、9次运行的训练轮数，并重新汇总已保存的实验指标。它不等同于重新训练模型。

安装 `requirements.txt` 中的依赖后，可运行 `python scripts/demo.py` 和 `python -m unittest discover -s tests -v`。示例使用合成数据，与论文实验明确区分。

## 实验与复现

- 数据为MUUFL Gulfport Campus 1；原始数据由[数据集作者仓库](https://github.com/GatorSense/MUUFLGulfport)获取，准备方法见[数据说明](data/README.md)。
- 三种空间划分 × 三个训练种子，共9次；每次固定训练30轮，训练标签模拟替换概率0.40。
- `python scripts/evaluate_checkpoints.py --device cuda --output-root reproduced_evaluation`：用已发布权重重新预测，需先准备数据。
- `python scripts/run_diagnostics.py --output-root reproduced_evaluation`：完成全部9次预测后，运行固定参数网格与空间评分对照。
- `python scripts/run_experiment.py --device cuda --output-root reproduced_training`：从头训练全部9次实验。
- `Rscript scripts/plot_paper_figures.R`：使用已保存的真实CSV数据绘图，需安装英文README列出的R包。

论文主要结果：平均覆盖率由88.87%升至92.96%，各次实验最低测试块覆盖率的均值由78.61%升至89.02%，平均预测集大小由1.0283升至1.2279。两种校准规则使用相同冻结概率，不改变argmax分类结果。

完整环境、方法归属、输入支持隔离、阈值计算、指标解释和复现步骤见[英文说明](README.md)及[复现指南](docs/REPRODUCIBILITY.md)。本仓库没有发布论文全文，也没有赋予论文录用状态。
