### 3D unet 靶区分割

## 模型配置文件
使用的是 `models/base_model/params.json`，按照交叉验证，把输入的总数据集平分为4个数据组，`train_batch`和`test_batch`分别表示训练集和测试集使用的数据组序号，使用不同的训练集修改序号即可。

## 运行
训练并预测输出
```bash
 python src/main.py -model_dir models/base_model -mode train_pred
```
仅训练
```bash
 python src/main.py -model_dir models/base_model -mode train
```
仅预测
```bash
 python src/main.py -model_dir models/base_model -mode pred
```

## 结果保存
按照params.json指定的训练集组序号，会在`models/base_model/`下创建新的目录，由训练集组序号连接下划线命名，如`0_1_2`，训练模型的ckpt文件会保存在这个目录，预测结果保存在其`/predict`子目录
