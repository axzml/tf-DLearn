# Logistic Regression

对本目录下的文件做一些说明:

+ `mnist_train.py`: LR 的基本实现, 使用 mnist 数据集作为输入, 验证训练代码的正确性.
+ `sparse_emb_train.py`: 更复杂一些, 读取自定义的数据集, 对于数据读取的优化可以参考 [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance);
还有对类别特征进行哈希之类的.


周末愉快！
**TODO**:

+ 增加测试数据集的读取代码
+ 增加测试代码, 比如计算 Accuracy, AUC 等等
+ 模型加载和保存
+ 日志保存
+ TensorBoard 可视化
+ GPU 支持
+ 分布式训练
