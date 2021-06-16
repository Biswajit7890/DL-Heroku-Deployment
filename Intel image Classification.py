#!/usr/bin/env python
# coding: utf-8

# In[26]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[27]:


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), kernel_initializer='TruncatedNormal',input_shape = (32, 32, 3), activation = 'selu'))
classifier.add(Conv2D(32, (3, 3),kernel_initializer='TruncatedNormal', activation = 'relu'))
classifier.add(Conv2D(32, (3, 3),kernel_initializer='TruncatedNormal', activation = 'selu'))
classifier.add(Conv2D(32, (3, 3),kernel_initializer='TruncatedNormal', activation = 'relu'))
classifier.add(Conv2D(32, (3, 3),kernel_initializer='TruncatedNormal', activation = 'selu'))
classifier.add(Conv2D(32, (3, 3),kernel_initializer='TruncatedNormal', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[28]:


classifier.add(Flatten())


# In[29]:



classifier.add(Dense(units = 100,kernel_initializer='TruncatedNormal', activation = 'relu'))
classifier.add(Dense(units = 30,kernel_initializer='TruncatedNormal', activation = 'relu'))
classifier.add(Dense(units = 6,kernel_initializer='TruncatedNormal', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[30]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,rotation_range=0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[31]:


train_path='E:\I neuron DLNLPCV\DL DATASETS\intel image classification\seg_train\seg_train'
test_path='E:\I neuron DLNLPCV\DL DATASETS\intel image classification\seg_test\seg_test'


# In[32]:


training_set = train_datagen.flow_from_directory(train_path,
target_size = (32, 32),batch_size = 32,class_mode = 'categorical')


# In[33]:


test_set = test_datagen.flow_from_directory(test_path,
target_size = (32, 32), batch_size = 32,class_mode = 'categorical')


# In[34]:


model = classifier.fit(training_set,steps_per_epoch = 1000,epochs = 20,validation_data = test_set,validation_steps = 500)


# In[35]:


model.history


# In[36]:


classifier.save("Intel_Image.h5")


# In[37]:


training_set.class_indices


# In[ ]:




