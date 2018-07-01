
# coding: utf-8

# # 快速上手Keras
# by [@寒小阳](http://blog.csdn.net/han_xiaoyang)

# ## 0.引言

# 我们今天的主角是keras，这是一个让很多同学爱不释手的深度学习框架，这是一个极易上手的框架，搭建神经网络就像搭积木一样简单，再也没有“卷积神经网络一层5元，循环神经网络一层10元...”的烦恼了。
# 
# 而Keras是一个站在巨人肩膀上的深度学习框架，其后端可以用不同的DL框架支撑，比如theano，比如Tensorflow，比如微软的CNTK。好消息是，原本被吐槽的速度问题，在使用CNTK做后端之后，官方测试表明甚至可以比原生态的Tensorflow速度还要快。
# 
# OK，所以主角来了。
# 
# ![](./images/keras.png)

# ## 1.各式各样的神经网络
# 我们先来看一张图，里面涵盖了各位同学关心的各种类型的神经网络。
# ![](./images/nns.png)

# ## 2.快速上手Keras
# 所以大家准备好搭积木了吗，轻松愉快的神经网络搭建学习马上开始。<br>
# 最常见的一类神经网络模型叫做Sequential/序贯模型，就是大家熟知的一层一层搭网络的这种模型过程。

# ### 2.1 引入库，初始化“模型架子”

# In[ ]:


from keras.models import Sequential
model = Sequential()


# ### 2.2 通过add来添加层
# 我们举一个最简单的MLP例子，这下面我们添加的都是全连接层

# In[ ]:


from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))


# ### 2.3 通过compile来编译模型

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# 编译模型时必须指明损失函数和优化器，如果你需要的话，也可以自己定制损失函数。Keras里也封装好了很多优化器和损失函数。

# In[ ]:


from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))


# ### 2.4 把数据灌进来训练

# In[ ]:


model.fit(x_train, y_train, epochs=5, batch_size=32)


# 你也可以选择，手动一批一批数据训练，大概就是下面这个样子

# In[ ]:


model.train_on_batch(x_batch, y_batch)


# ### 2.5 在测试集上评估效果

# In[ ]:


loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)


# ### 2.6 实际预测

# In[ ]:


classes = model.predict(x_test, batch_size=128)

