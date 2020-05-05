import os

import tensorflow as tf
from tensorflow import keras
from util import AttackUtil

from cleverhans.dataset import MNIST
from cleverhans.utils_keras import cnn_model, KerasModelWrapper

# Setting up training parameters
tf.set_random_seed(4557077)

learning_rate = 1e-4

label_smoothing = 0.1

log_file = "df1500.txt"
accuracies_log_file = "accuracies/" + log_file

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with sess.as_default():
    # Get MNIST test data
    mnist = MNIST(train_start=0, train_end=60000,
                  test_start=0, test_end=10000)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    # split test data to test and evaluation samples
    x_evaluate, y_evaluate = x_test[6000:], y_test[6000:]
    x_test, y_test = x_test[0:6000], y_test[0:6000]

    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Label smoothing
    y_train -= label_smoothing * (y_train - 1. / nb_classes)

    # Define Keras model
    model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                      channels=nchannels, nb_filters=64,
                      nb_classes=nb_classes)

    wrap = KerasModelWrapper(model)

    attack_util = AttackUtil(wrap, sess, x_evaluate, y_evaluate, log_file)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # initialize model gradients
    model.fit(x_train, y_train,
              batch_size=64,
              epochs=1,
              validation_data=(x_test, y_test),
              verbose=2)

    # training loop
    for epoch in range(1500):
        print(epoch)
        X_adv, Y_adv = attack_util.deep_fool_util.create_adversaries(x_train, y_train, epoch, 1500, overshoot=-0.02)
        model.fit(X_adv, Y_adv,
                  batch_size=128,
                  epochs=1,
                  validation_data=(x_test, y_test),
                  verbose=2)
        if epoch % 10 == 0:
            attack_util.log_current_accuracies(model, accuracies_log_file)

