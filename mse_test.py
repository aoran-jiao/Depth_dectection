import os
import tensorflow as tf
from PIL import Image
import numpy as np
import math
import random
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

orw, orh = 4160,2340
pixw,pixh = 256,144


inputs = []
labels = []

base = os.path.dirname(os.path.abspath("__file__"))
img_dir = os.path.join(base,"dispic")

category_size = 7

interval = 0.5
critical = interval*(category_size-1)

target_after_aug = [2.25,2.75]

coefd = {0:0.084,1:0.18,2:0.081,3:0.064, 4:0.064, 5: 0.064 }
mutid = {0:2, 1:2, 2:3, 3:3, 4:3, 5:3}
k = 20
p = 0
for r,d,files in os.walk(img_dir):
    print(r)

    for f in files:
        p +=1
        if p == k:
            if f.endswith("png") or f.endswith("jpg"):

                path = os.path.join(r,f)
                op_image = Image.open(path)
                img = op_image.convert("L")
                op_image.close()

                label = int(f[:-4:])/1000

                base_cat = int(label/interval) 
                max_cat = min(base_cat, category_size-1)

                if max_cat< (category_size-1):

                    for rotate in range(0,1):
                        angle = int(rotate *3)
                        new_rotate = img.rotate(angle)
                        coef = coefd[max_cat]
#                         
                        for cp in range(mutid[max_cat],(abs(rotate)-1),-1):
                            minimize = (1-coef*cp)
                            cpw, cph = int(orw*minimize),int(orh*minimize)
                            stpw, stph = int((orw-cpw)/2),int((orh-cph)/2)
                            cp_img = new_rotate.crop((stpw, stph, stpw+cpw , stph+cph ))
                            cp_img.resize((pixw,pixh))

                            img1 = np.array(cp_img)

                            img2 = np.flip(img1,0)

                            img3 = np.flip(img1,1)

                            img4 = np.flip(img1)

                            real_label = round(minimize**2*label, 3)
                            print(real_label, end = " ")

                            for im in [img1]:
                                get_im = Image.fromarray(im).resize((256,144))
                                get_im.show()
                                broken = np.array(get_im)/255
                                scaled = scaler.fit_transform(broken)
                                mod = (scaled).reshape(144,256)
                               
                                inputs.append(mod)
                                labels.append(real_label)      
#                         break
#                     break


                else:
                    for taa in target_after_aug:
                        random_factor = random.randint(-300,300)
                        add_f = random_factor*0.001
                        taa += add_f
                        times = 2
                        the_coef = round((1-math.sqrt(taa/label))/times,3)
                        
                        for rotate in range(-2,3):
                            angle = int(rotate *6)
                            new_rotate = img.rotate(angle)
                            coef = the_coef
                            for cp in range(times, (abs(rotate)-1),-1):
                                minimize = (1-coef*cp)
                                cpw, cph = int(orw*minimize),int(orh*minimize)
                                stpw, stph = int((orw-cpw)/2),int((orh-cph)/2)
                                cp_img = new_rotate.crop((stpw, stph, stpw+cpw , stph+cph ))
                                cp_img.resize((pixw,pixh))

                                img1 = np.array(cp_img)

                                img2 = np.flip(img1,0)

                                img3 = np.flip(img1,1)

                                img4 = np.flip(img1)

                                real_label = round(minimize**2*label, 3)
                                print(real_label, end = " ")

                                for im in [img1, img2, img3, img4]:
                                    get_im = Image.fromarray(im).resize((256,144))
                                    get_im.show()
                                    broken = np.array(get_im)/255
                                    scaled = scaler.fit_transform(broken)
                                    mod = (scaled).reshape(144,256)
                                    inputs.append(mod)
                                    labels.append(real_label)



            
print(critical)

inputs = np.array(inputs).reshape(-1,pixh,pixw,1)
labels = np.array(labels)

input_size = inputs.shape[0]
np.random.seed(666)
indexes = np.random.permutation(input_size)

base = os.path.dirname(os.path.abspath("__file__"))
img_dir = os.path.join(base,"dispic")

category_size = 7

interval = 0.5
critical = interval*(category_size-1)

target_after_aug = [2.25,2.75]

coefd = {0:0.084,1:0.18,2:0.081,3:0.064, 4:0.064, 5: 0.064 }
mutid = {0:2, 1:2, 2:3, 3:3, 4:3, 5:3}
inputs = np.load('Memory/reginputs.npy')
labels = np.load('Memory/reglabels.npy')
# np.('Memory/reglabels.npy',labels)
# print(inputs.shape)
# print(labels.shape)
# np.save('Memory/reglabels.npy',)
# lbs = np.array(list(y_train).extend(list(y_test)))
# lbs.shape

cd = round((1-math.sqrt(2.1/14))/6,3)
mt = 6
num = 12
print((num*(1-cd*mt)*(1-cd*mt))," ", min((num/0.5), 6))
# print((1-0.04*2)**2*3.16)
# print(round((1-math.sqrt(2.1/14))/6,3))

inputs = np.array(inputs).reshape(-1,pixh,pixw,1)
labels = np.array(labels)

input_size = inputs.shape[0]
np.random.seed(666)
indexes = np.random.permutation(input_size)
labels.reshape(-1,1)

for kt in range(6):
    c = 0
    removes = []
    for i in range(labels.shape[0]):
        if np.argmax(labels[i]) == kt:
            removes.append(i)
    print(len(removes))
input_size = inputs.shape[0]
np.random.seed(666)
indexes = np.random.permutation(input_size)
# removes = removes[:000]
# inputs = np.delete(inputs,removes,0)
# labels = np.delete(labels,removes,0)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test =  train_test_split(inputs, labels, test_size = 0.3) 
# size = x_train.shape[0]
# test_size = x_test.shape[0]

# kk = np.array([k for k in range(143)])
# kk[kk[int(143*0.7)::]]
# # kk[indexes[:int(input_size*0.7):]]
# inputs = np.array(inputs).reshape(-1,pixh,pixw,1)

## remove overbalanced data please
labels = labels.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=666)
size = x_train.shape[0]
print(x_train.shape)
print(y_train.shape)

test_inputs = []
test_labels = []

base = os.path.dirname(os.path.abspath("__file__"))
img_dir = os.path.join(base,"test")

for r,d,files in os.walk(img_dir):
    print(r)

    for f in files:
        if f.endswith("png") or f.endswith("jpg"):

            path = os.path.join(r,f)
            op_image = Image.open(path)
            img = op_image.convert("L")
            op_image.close()
                
            label = int(f[:-4:])/1000
            
            base_cat = int(label/interval) 
            max_cat = min(base_cat, category_size-1)
            
            if max_cat< (category_size-1):
            
                for rotate in range(1):
                    angle = int(rotate *3)
                    new_rotate = img.rotate(angle)
                    coef = coefd[max_cat]
                    for cp in range(1):
                        minimize = (1-coef*cp)
                        cpw, cph = int(orw*minimize),int(orh*minimize)
                        stpw, stph = int((orw-cpw)/2),int((orh-cph)/2)
                        cp_img = new_rotate.crop((stpw, stph, stpw+cpw , stph+cph ))
                        cp_img.resize((pixw,pixh))

                        img1 = np.array(cp_img)

#                         img2 = np.flip(img1,0)

#                         img3 = np.flip(img1,1)

#                         img4 = np.flip(img1)

                        real_label = round(minimize**2*label,3)
                        print(real_label, end = " ")


#                         zl = [0]*category_size
#                         pick = min(int(real_label/interval), category_size-1)
#                         print(pick, end = "")
#                         zl[pick] = 1
#                         real_label = zl


                        for im in [img1]:
                            get_im = Image.fromarray(im).resize((256,144))
                            broken = np.array(get_im)/255
                            scaled = scaler.fit_transform(broken)
                            mod = (scaled).reshape(144,256)
                            test_inputs.append(mod)
                            test_labels.append(real_label)
                            
                            
                            
                            
                            
            else:
                for taa in target_after_aug:
                    random_factor = random.randint(-20,20)
                    add_f = random_factor*0.01
                    taa += add_f
                    times = 3
                    the_coef = round((1-math.sqrt(taa/label))/times,3)

                    for rotate in range(1):
                        angle = int(rotate *3)
                        new_rotate = img.rotate(angle)
                        coef = the_coef
                        for cp in range(1):
                            minimize = (1-coef*cp)
                            cpw, cph = int(orw*minimize),int(orh*minimize)
                            stpw, stph = int((orw-cpw)/2),int((orh-cph)/2)
                            cp_img = new_rotate.crop((stpw, stph, stpw+cpw , stph+cph ))
                            cp_img.resize((pixw,pixh))

                            img1 = np.array(cp_img)

#                             img2 = np.flip(img1,0)

#                             img3 = np.flip(img1,1)

#                             img4 = np.flip(img1)
                            
#                             real_label = round(minimize**2*label,3)
#                             print(real_label, end = " ")
# #                             zl = [0]*category_size
# #                             pick = min(int(real_label/interval), category_size-1)
# #                             print(pick, end = "")
# #                             zl[pick] = 1
# #                             real_label = zl

                            
                            for im in [img1]:
                                get_im = Image.fromarray(im).resize((256,144))
                                broken = np.array(get_im)/255
                                scaled = scaler.fit_transform(broken)
                                mod = (scaled).reshape(144,256)
                                test_inputs.append(mod)
                                test_labels.append(real_label)


                
            
print(critical)

# # unzip image
# show_sample = ((inputs[4].reshape(-1)+1)*127.5).astype(np.int).reshape(pixh,pixw)
# new_im = Image.fromarray(show_sample)
# new_im.show()
# print(inputs.shape)


test_inputs = np.array(test_inputs).reshape(-1,pixh,pixw,1)
test_labels = np.array(test_labels)

test_input_size = test_inputs.shape[0]

import os
import tensorflow as tf
from PIL import Image
import numpy as np
orw, orh = 4160,2340
pixw,pixh = 256,144




n_inputs = pixw*pixh
input_width,input_height = pixw, pixh
input_channels = 1  

n_hidden_in = 16*9*512
n_outputs = 1*1*1*1


training = tf.placeholder_with_default(True, shape = (),name = "tra")


X = tf.placeholder(tf.float32, [None,pixh,pixw,1])
y = tf.placeholder(tf.int32,shape = [None,1]) 

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('mnist',one_hot = True)
# input_x = tf.placeholder(tf.float32, [None,28*28])/255
# X = tf.reshape(input_x,[-1,28,28,1])
# y = tf.placeholder(tf.int32, [None,10])
with tf.name_scope('Model'):
    with tf.name_scope('CNN'):
        coin = tf.contrib.layers.xavier_initializer(uniform=False)
        regul = tf.contrib.layers.l2_regularizer(scale=0.1)


        conv10 = tf.layers.conv2d(X, 64, [5,5], 1, "same",activation = tf.nn.relu, kernel_initializer = coin)
        conv21 = tf.layers.conv2d(conv10, 128, [3,3], 1, "same",activation = tf.nn.relu, kernel_initializer = coin)
        pool1 = tf.layers.max_pooling2d(conv21, [2,2], 2)

        conv22 = tf.layers.conv2d(pool1, 128, [3,3], 1, "same",activation = tf.nn.relu, kernel_initializer = coin)
        conv23 = tf.layers.conv2d(conv22, 128, [3,3], 2, "same",activation = tf.nn.relu, kernel_initializer = coin)
        bn2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv23, training = training, momentum = 0.9))
        pool2 = tf.layers.max_pooling2d(bn2, [2,2], 2)


        conv31 = tf.layers.conv2d(pool2, 256, [3,3], 1, "same",activation = tf.nn.relu, kernel_initializer = coin)
        conv32 = tf.layers.conv2d(conv31, 512, [3,3], 1, "same",activation = tf.nn.relu, kernel_initializer = coin)
        bn3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv32, training = training, momentum = 0.9))
        pool3 = tf.layers.max_pooling2d(bn3, [2,2], 2)

        # conv41 = tf.layers.conv2d(pool3, 1024, [3,3], 1, "same",activation = tf.nn.relu, kernel_initializer = coin)
        # conv42 = tf.layers.conv2d(conv41, 1024, [3,3], 1, "same",activation = tf.nn.relu, kernel_initializer = coin)
        # bn4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv41, training = training, momentum = 0.9))
        # pool4 = tf.layers.max_pooling2d(bn4, [2,2], 2)

    with tf.name_scope('Fully_Connected'):
        with tf.name_scope('Flatten'):
            flat = tf.reshape(pool3, shape = [-1,n_hidden_in])

        hidden1 = tf.layers.dense(flat,2048,kernel_regularizer = regul , activation = tf.nn.relu, kernel_initializer = coin)
        hidden1_bn = tf.nn.leaky_relu(tf.layers.batch_normalization(hidden1, momentum = 0.9))
        X_drop = tf.layers.dropout(inputs = hidden1_bn, rate = 0.5, training = training)


        hidden2 = tf.layers.dense(X_drop,2048,kernel_regularizer = regul,  activation =tf.nn.relu, kernel_initializer = coin)
        hidden2_bn =  tf.nn.leaky_relu(tf.layers.batch_normalization(hidden2, momentum = 0.9))

        X_drop0 = tf.layers.dropout(inputs = hidden2_bn, rate = 0.5, training = training)

    with tf.name_scope('Outputs'):
        y_pred = tf.layers.dense(X_drop0, units = 1)
    # Y_proba = tf.nn.softmax(y_pred)

with tf.name_scope('Loss'):

    loss = tf.losses.mean_squared_error(labels = y,predictions = y_pred)
tf.summary.scalar('Loss', loss)
# loss = tf.reduce_mean(xentropy)


# acc = tf.metrics.accuracy(labels=tf.argmax(y, 1), 
#                                   predictions=tf.argmax(y_pred,1))[1]
# accuracy = tf.metrics.accuracy(
#             labels = tf.argmax(y,axis = 1),
#             predictions = tf.argmax(y_pred,axis = 1)
#             ) [1]
with tf.name_scope('train_gradients'):

    global_step = tf.Variable(0,trainable = False, name = "global_")

    learning_rate = tf.placeholder(tf.float32)
    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True)

    threshold = 1
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(grad if grad is None else tf.clip_by_value(grad,-threshold, threshold),
                   var) for grad, var in gvs]

merged = tf.summary.merge_all()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    training_op = optimizer.apply_gradients(capped_gvs,global_step = global_step)

init2 = tf.local_variables_initializer()
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())   
saver = tf.train.Saver()

# with tf.device('/gpu:0'):

# import sklearn
n_epochs = 200
batch_size = 40
x_train_shape = x_train.shape[0]
steps = x_train_shape//batch_size

lr = 0.0001
config = tf.ConfigProto(intra_op_parallelism_threads = 1 , inter_op_parallelism_threads =1 )
Bool = bool(True)
with tf.Session() as sess:
    if os.path.isfile("regret checkpoint/my_model_final.ckpt.index"):
        saver.restore(sess, "regret checkpoint/my_model_final.ckpt")
        init2.run()
    else:
        train_writer = tf.summary.FileWriter('./regretlog/train', sess.graph)
        test_writer = tf.summary.FileWriter('./regretlog/test')
        init.run()

    for epoch in range(n_epochs):
        shuffle = np.random.permutation(x_train.shape[0])
        for i in range(steps):
            indeces = shuffle[batch_size*i:batch_size*(i+1):]
            X_batch = x_train[indeces]
            y_batch = y_train[indeces]
            sess.run(update_ops, feed_dict = {X: X_batch, y: y_batch,training: Bool, learning_rate: lr})

            if i % 120 == 0: 
                t_shuffle = np.random.permutation(x_test.shape[0])
                t_indexes = [j for j in range(50)]
                x_test_b = x_test[t_shuffle[t_indexes]]
                y_test_b = y_test[t_shuffle[t_indexes]]
                
                summary, cost = sess.run([merged,loss], feed_dict={X: x_test_b, y: y_test_b,training: False })
                
                test_writer.add_summary(summary, global_step.eval())
                print('Loss at step %s is now %s' % (global_step.eval(), cost))
                print("save")
                save_path = saver.save(sess,"regret checkpoint/my_model_final.ckpt")
               
                
            elif i% 50:  
                summary, _ = sess.run([merged, training_op],  feed_dict = {X: X_batch, y: y_batch,training: Bool, learning_rate: lr})
                train_writer.add_summary(summary, global_step.eval())
        
with tf.Session() as sess:
    if os.path.isfile("regret checkpoint/my_model_final.ckpt.index"):
        saver.restore(sess, "regret checkpoint/my_model_final.ckpt")
        init2.run()
#         training = tf.placeholder_with_default(False, shape = (),name = "training")
        t_shuffle = np.random.permutation(test_inputs.shape[0])
        t_indexes = [i for i in range(10)]
        x_test_b = test_inputs[t_shuffle[t_indexes]]
        y_test_b = test_labels[t_shuffle[t_indexes]]

        test_output = sess.run(y_pred,{X: x_test_b, training:True})
        inferenced_y = np.round(test_output.reshape(-1),decimals = 3)
        print(inferenced_y,'Inferenced numbers')
        print(y_test_b.reshape(-1),'real number')
#         print(acc.eval( feed_dict = {X: x_test_b, y: y_test_b,training: False}))

