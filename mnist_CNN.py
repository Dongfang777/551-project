import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#读取经典的MNIST数据集
#使用one-hot独热码，每个稀疏向量只有标签类值是1，其他类是0
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def show_result(images,labels,prediction,index,num=10):     #绘制图形显示预测结果
    pic=plt.gcf()                                           #获取当前图像
    pic.set_size_inches(10,12)                              #设置图片大小
    for i in range(0,num):
        sub_pic=plt.subplot(5,5,i+1)                        #获取第i个子图
        #将第index个images信息显示到子图上
        sub_pic.imshow(np.reshape(images[index],(28,28)),cmap='binary') 
        title="label:"+str(np.argmax(labels[index]))        #设置子图的title内容
        if len(prediction)>0:
            title+=",predict:"+str(prediction[index])
            
        sub_pic.set_title(title,fontsize=10)
        sub_pic.set_xticks([])                              #设置x、y坐标轴不显示
        sub_pic.set_yticks([])
        index+=1
    plt.show()
#输入数据占位符
x = tf.placeholder('float', [None, 784])
#数据标签占位符
y_ = tf.placeholder('float', [None, 10])
# 输入图片数据形状转化为28*28矩阵
x_image = tf.reshape(x, [-1, 28, 28, 1])
#第一层卷积层，初始化卷积核参数、偏置值，该卷积层5*5大小，1个通道
#共有6个不同卷积核，步长为1*1，等大填充
filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
bias1 = tf.Variable(tf.truncated_normal([6]))
conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.sigmoid(conv1 + bias1)
#第一层池化层，采用最大值池化，尺寸为2*2，步长为2*2，等大填充
maxPool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#第二层卷积层，初始化卷积核参数、偏置值，该卷积层5*5大小，6个输入通道
#共有16个不同卷积核，步长为1*1，等大填充
filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
bias2 = tf.Variable(tf.truncated_normal([16]))
conv2 = tf.nn.conv2d(maxPool2, filter2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.sigmoid(conv2 + bias2)
#第二层池化层，采用最大值池化，尺寸为2*2，步长为2*2，等大填充
maxPool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#第三层卷积层，初始化卷积核参数、偏置值，该卷积层5*5大小，16个输入通道
#共有120个不同卷积核，步长为1*1，等大填充
filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]))
bias3 = tf.Variable(tf.truncated_normal([120]))
conv3 = tf.nn.conv2d(maxPool3, filter3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3 = tf.nn.sigmoid(conv3 + bias3)

# 全连接层
# 产生权值参数、偏置变量
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]))
b_fc1 = tf.Variable(tf.truncated_normal([80]))
# 形状转化，将卷积的产出展开
h_pool2_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])
# 神经网络计算，并添加sigmoid激活函数
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 输出层，使用softmax进行多分类
# 产生权值参数、偏置变量
W_fc2 = tf.Variable(tf.truncated_normal([80, 10]))
b_fc2 = tf.Variable(tf.truncated_normal([10]))
# 神经网络计算
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
# 损失函数

train_epochs=50
batch_size=100                                                #每个批次的样本数
batch_num=int(mnist.train.num_examples/batch_size)            #一轮需要训练多少批
learning_rate=0.01

train_loss = []
train_acc = []
test_losss = []
test_accs = []

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 使用GDO优化算法来调整参数
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

ss=tf.Session()
init=tf.global_variables_initializer()
ss.run(init)

for epoch in range(train_epochs):
    for batch in range(batch_num):                            #分批次读取数据进行训练
        xs,ys=mnist.train.next_batch(batch_size)
        ss.run(optimizer,feed_dict={x:xs,y_:ys})
    #每轮训练结束后通过带入验证集的数据，检测模型的损失与准去率 
    loss,acc=ss.run([cross_entropy,accuracy],feed_dict={x:mnist.train.images,y_:mnist.train.labels})
    test_loss,test_acc=ss.run([cross_entropy,accuracy],feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    print('第%2d轮训练：损失为：%9f，训练准确率：%.4f'%(epoch+1,loss,acc),'第%2d轮训练：损失为：%9f，测试准确率：%.4f'%(epoch+1,test_loss,test_acc))
    train_loss.append(loss)
    train_acc.append(acc)
    test_losss.append(test_loss)
    test_accs.append(test_acc)

prediction=ss.run(tf.argmax(y_conv,1),feed_dict={x:mnist.test.images})

train_loss_ = np.array(train_loss)
train_acc_ = np.array(train_acc)
test_losss_ = np.array(test_losss)
test_accs_ = np.array(test_accs)

x = np.linspace(1,50,50)
l1, = plt.plot(x,train_loss_)
#l2, = plt.plot(x,validation_acc_)
l3, = plt.plot(x,test_losss_)
#l4, = plt.plot(x,test_accs_)

plt.legend(handles=[l1, l3], labels=['Train_loss', 'Test_loss'],  loc='best')
#plt.xticks(x) #这是为了限制坐标轴显示为整数
plt.xlabel("Number of epochs")
plt.ylabel("Loss") 
plt.show()

l2, = plt.plot(x,train_acc_)
l4, = plt.plot(x,test_accs_)
plt.legend(handles=[l2, l4], labels=['Train_accuracy', 'Test_accuracy'],  loc='best')
#plt.xticks(x) #这是为了限制坐标轴显示为整数
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.show()

show_result(mnist.test.images,mnist.test.labels,prediction,10)

# 关闭会话
ss.close()
