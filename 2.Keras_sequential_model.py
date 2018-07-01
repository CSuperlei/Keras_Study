
# coding: utf-8

# # Keras序贯模型

# 跃跃欲试的宝宝们，你们刚才已经见到了用Keras搭建一个神经网络有多么精简。下面我们来看一类最简单，同时也是使用最多的神经网络结果，在Keras中我们使用序贯模型对它进行构建。
# 
# 序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”。

# ## 1.序贯模型要点

# ### 1.1 简单的汉堡堆叠 or 手动加菜
# 可以通过向Sequential模型传递一个layer的list来构造序贯模型，或者通过.add()方法一个个的将layer加入模型中

# In[ ]:


# 一次性构建汉堡
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
Dense(32, units=784),
Activation('relu'),
Dense(10),
Activation('softmax'),
])


# In[ ]:


# ”手动加菜“
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))


# ### 1.2 告诉模型数据的维度
# 模型需要知道输入数据的shape，Sequential的第一层需要接受一个关于输入数据shape的参数，后面的各个层则可以自动的推导出中间数据的shape。第一层的数据维度可以这么给进去(和Tensorflow定义placeholder有点像)：
# - 传递一个input_shape的关键字参数给第一层，input_shape是一个tuple类型的数据，其中也可以填入None，如果填入None则表示此位置可能是任何正整数。数据的batch大小不应包含在其中。
# - 有些2D层，如Dense，支持通过指定其输入维度input_dim来隐含的指定输入数据shape。一些3D的时域层支持通过参数input_dim和input_length来指定输入shape。
# - 如果你需要为输入指定一个固定大小的batch_size（常用于stateful RNN网络），可以传递batch_size参数到一个层中，例如你想指定输入张量的batch大小是32，数据shape是（6，8），则你需要传递batch_size=32和input_shape=(6,8)。

# In[ ]:


model = Sequential()
model.add(Dense(32, input_dim=784))


# In[ ]:


model = Sequential()
model.add(Dense(32, input_shape=784))


# ### 1.3 编译你的模型

# 在训练模型之前，我们需要通过compile来对学习过程进行配置。compile接收三个参数：
# 
# - **优化器optimizer**：该参数可指定为已预定义的优化器名，如rmsprop、adagrad，或一个Optimizer类的对象，详情见[optimizers](http://keras-cn.readthedocs.io/en/latest/other/optimizers/)
# - **损失函数loss**：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse，也可以为一个损失函数。详情见[losses](http://keras-cn.readthedocs.io/en/latest/other/objectives/)
# - **指标列表metrics**：对分类问题，我们一般将该列表设置为metrics=['accuracy']。指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.指标函数应该返回单个张量,或一个完成metric_name - > metric_value映射的字典.请参考[性能评估](http://keras-cn.readthedocs.io/en/latest/other/metrices/)

# In[ ]:


# 多分类问题
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 二分类问题
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 回归问题
model.compile(optimizer='rmsprop',
              loss='mse')

# 自定义metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])


# ### 1.4 训练

# 如果你的数据可以一次性读进内容进行建模，那么你会用到fit函数

# In[ ]:


# 构建与编译模型
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 查出数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# 训练与数据拟合
model.fit(data, labels, epochs=10, batch_size=32)


# 如果你的数据量很大，你可能要用到fit_generator

# In[ ]:


def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            x, y = process_line(line)
            img = load_images(x)
            yield (img, y)
        f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, nb_epoch=10)


# ## 2.一些序贯模型的例子

# ### 2.1 多层感知器

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)


# ### 2.2 卷积神经网络

# In[ ]:


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)


# ### 2.3 循环神经网络

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)

