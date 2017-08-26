import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np

# TODO: Load traffic signs data.
training_file = 'train.p'
#testing_file  = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
#with open(testing_file, mode='rb') as f:
    #test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
#X_test, y_test = test['features'], test['labels']

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# TODO: Define placeholders and resize operation.
nb_classes = 43
BATCH_SIZE = 128
EPOCHS     = 20

# 1-hot encode labels
y_train    = (np.arange(nb_classes) == y_train[:, None]).astype(np.float32)
y_val      = (np.arange(nb_classes) == y_val[:, None]).astype(np.float32)

x = tf.placeholder(tf.float32, (BATCH_SIZE, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
y = tf.placeholder(tf.float32, shape=(BATCH_SIZE,nb_classes))

loss   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.

def eval_data(images, labels):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    steps_per_epoch = images.shape[0] // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc = 0
    sess = tf.get_default_session()
    for step in range(steps_per_epoch):
        batch_start = step*BATCH_SIZE
        batch_x = images[batch_start:batch_start + BATCH_SIZE,:,:,:]
        batch_y = labels[batch_start:batch_start + BATCH_SIZE,:]
        
        acc = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
    return total_acc/num_examples
    
with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = X_train.shape[0] // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            
            # Random shuffle training data
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
            
            for step in range(steps_per_epoch):
                batch_start = step*BATCH_SIZE
                
                if (step%5) == 0:
                    print(step)
                
                batch_x     = X_train[batch_start:batch_start + BATCH_SIZE,:,:,:]
                batch_y     = y_train[batch_start:batch_start + BATCH_SIZE,:]
                
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        
            val_acc = eval_data(X_val, y_val)
            print("EPOCH {} ...".format(i+1))
            print("Validation accuracy = {:.3f}".format(val_acc))
        
        # Evaluate on the test data
        #test_acc = eval_data(X_test, y_test)
        #print("Test accuracy = {:.3f}".format(test_acc))
