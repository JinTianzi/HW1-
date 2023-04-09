# HW1-构建两层神经网络

### 直接运行“HW1 构建两层神经网络.ipynb”或者“HW1 构建两层神经网络.py”，就可以完成模型训练、调参和测试。

* 也可以单独进行模型训练和测试：使用model(x_train, y_train, x_test, y_test, learning_rate, hidden_size,lambda_,iters_num,batch_size)函数。
    * 输入：x_train, y_train, x_test, y_test，learning_rate（学习率），hidden_size（隐藏层大小），lambda_（L2正则化强度），iters_num（迭代次数），batch_size（一次训练所选取的样本数）。
    * 输出：train_loss_list（训练集loss列）,test_loss_list（测试集loss列）,train_acc_list（训练集accuracy列）,test_acc_list（测试集accuracy列）,W1（参数W1）,W2（参数W2）,b1（参数b1）,b2（参数b2）。
    * 同时会打印每个epoch训练和测试的loss和准确率。
* 直接运行para_search(x_train, y_train, learning_rate_set, hidden_size_set,lambda_set,iters_num,batch_size)函数可用于调参。
    * 输入：x_train, y_train，learning_rate_set（学习率参数选择范围），hidden_size_set（隐藏层大小参数选择范围），lambda_setL2（正则化强度的参数选择范围）。
    * 输出：调参选中的learning_rate（学习率）, hidden_size（隐藏层大小）,lambda_（L2正则化强度）。
    * 同时会打印每种参数组合下，模型训练过程中每个epoch训练和测试的loss和准确率。
