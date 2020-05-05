from datetime import datetime

import numpy
import tensorflow as tf

from cleverhans.attacks import DeepFool, ProjectedGradientDescent
from cleverhans.attacks.deep_fool import deepfool_batch

class DeepFoolUtil(object):

    def __init__(self, model, sess, log_file="log.txt"):
        self.model = model
        self.sess = sess
        self.log_file = log_file

    def create_adversaries(self, x_train, y_train, i, nb_of_adv=None, nb_candidate=10, overshoot=0.02, max_iter=50, clip_min=0., clip_max=1.):
        """
        :param x_train: numpy array of test data.
        :param y_train: numpy array of test classifications.
        :param i: current number of epoch
        :param nb_of_adv: number of needed adversarial examples
        Returns training set with containing nb_of_examples deep fool adversarial examples. The examples to be perturbed
        is chosen by the current epoch number.
        """

        if nb_of_adv is None:
            nb_of_adv = len(x_train)

        adv_train_x, clean_train_x, adv_train_y, clean_train_y = divide_into_clean_and_adversarial_set(x_train, y_train, nb_of_adv, i)

        adv_x = self.attack_in_parts(adv_train_x, nb_candidate, overshoot, max_iter, clip_min, clip_max)

        return numpy.vstack((adv_x, clean_train_x)), numpy.vstack((adv_train_y, clean_train_y))

    def attack_in_parts(self, adv_train_x, nb_candidate, overshoot, max_iter, clip_min, clip_max):
        adv_x = None
        if len(adv_train_x) <= 10000:
            x = tf.placeholder(tf.float32, shape=adv_train_x.shape)
            adv_x = self.deepfool_batch(
                x, adv_train_x, nb_candidate, overshoot, max_iter, clip_min, clip_max)
        else:
            # it is needed because of memory boundries and now handles only multiple of 10 000 samples
            parts = int(len(adv_train_x) / 10000)
            for slice in range(parts):
                adv_train_x_part = adv_train_x[slice * 10000:(slice + 1) * 10000]
                x = tf.placeholder(tf.float32, shape=adv_train_x_part.shape)
                adv_x_part = self.deepfool_batch(
                    x, adv_train_x_part, nb_candidate, overshoot, max_iter, clip_min, clip_max)
                if adv_x is None:
                    adv_x = adv_x_part
                else:
                    adv_x = numpy.vstack((adv_x, adv_x_part))

        return adv_x

    def deepfool_batch(self,
                       x,
                       X,
                       nb_candidate,
                       overshoot,
                       max_iter,
                       clip_min,
                       clip_max):
        logits = self.model.get_logits(x)
        preds = tf.reshape(
            tf.nn.top_k(logits, k=nb_candidate)[0],
            [-1, nb_candidate])
        nb_classes = logits.get_shape().as_list()[-1]

        from cleverhans.utils_tf import jacobian_graph
        grads = tf.stack(jacobian_graph(preds, x, nb_candidate), axis=1)

        return deepfool_batch(self.sess, x, preds, logits, grads, X,
                              nb_candidate, overshoot,
                              max_iter, clip_min, clip_max, nb_classes)


class PGDUtil(object):

    def __init__(self, model, sess, log_file="log.txt"):
        self.pgd = ProjectedGradientDescent(model=model, sess=sess)
        self.log_file = log_file

    def create_adversaries(self, x_train, y_train, i, nb_of_adv=None):
        if nb_of_adv is None:
            nb_of_adv = len(x_train)

        adv_train_x, clean_train_x, adv_train_y, clean_train_y = divide_into_clean_and_adversarial_set(x_train, y_train, nb_of_adv, i)
        adv_x = self.pgd.generate_np(adv_train_x)
        return numpy.vstack((adv_x, clean_train_x)), numpy.vstack((adv_train_y, clean_train_y))


def divide_into_clean_and_adversarial_set(x_train, y_train, nb_of_adv, i):
    clean_x_train_postfix, clean_y_train_postfix = [], []
    clean_x_train_suffix, clean_y_train_suffix = [], []
    index = int(i % (len(x_train) / nb_of_adv))
    if index is not 0:
        clean_x_train_postfix, clean_y_train_postfix = x_train[0: index * nb_of_adv], y_train[0: index * nb_of_adv]
    adv_train_x, adv_train_y = x_train[index * nb_of_adv:(index + 1) * nb_of_adv], y_train[index * nb_of_adv:(index + 1) * nb_of_adv]
    if index is not len(x_train) / nb_of_adv - 1:
        clean_x_train_suffix, clean_y_train_suffix = x_train[(index + 1) * nb_of_adv:], y_train[(index + 1) * nb_of_adv:]

    if len(clean_x_train_postfix) > 0:
        if len(clean_x_train_suffix) > 0:
            clean_train_x = numpy.vstack((clean_x_train_postfix, clean_x_train_suffix))
            clean_train_y = numpy.vstack((clean_y_train_postfix, clean_y_train_suffix))
        else:
            clean_train_x = clean_x_train_postfix
            clean_train_y = clean_y_train_postfix
    else:
        clean_train_x = clean_x_train_suffix
        clean_train_y = clean_y_train_suffix

    return adv_train_x, clean_train_x, adv_train_y, clean_train_y


def save_perturbed_data(x_adv, log_file):
    numpy.save(log_file, x_adv)


#   maximum, minimum, average, median = euclidean_distance_metrics(x, x_adv)
#   with open(log_file, "a") as f:
#       f.write('Evaluate perturbation:\n'.format(datetime.now()))
#       f.write('Minimum perturbation: {}\n'.format(minimum))
#       f.write('Maximum perturbation: {}\n'.format(maximum))
#       f.write('Average perturbation: {}\n'.format(average))
#       f.write('Median of perturbations: {}\n\n'.format(median))


def log_acc_and_loss(acc, loss, log_file, eval_desc="without"):
    with open(log_file, "a") as f:
        f.write('Evaulate {} attack:\n'.format(eval_desc))
        f.write('Accuracy: {}\n'.format(acc))
        f.write('Loss: {}\n\n'.format(loss))


class AttackUtil(object):

    def __init__(self, model, sess, x_evaluate, y_evaluate, log_file):
        self.model = model
        self.sess = sess
        self.deep_fool_util = DeepFoolUtil(model, sess, log_file)
        self.pgd_util = PGDUtil(model=model, sess=sess, log_file=log_file)
        self.log_file = log_file
        self.x_evaluate = x_evaluate
        self.y_evaluate = y_evaluate

    def log_current_accuracies(self, model, log_file, batch_size=128):
        loss_value, acc_clean = model.evaluate(self.x_evaluate, self.y_evaluate, batch_size)

        x_adv = self.pgd_util.pgd.generate_np(self.x_evaluate)
        loss_value, acc_pgd = model.evaluate(x_adv, self.y_evaluate, batch_size)

        x_adv = self.deep_fool_util.create_adversaries(self.x_evaluate, self.y_evaluate, 0, max_iter=100)
        loss_value, acc_df = model.evaluate(x_adv, self.y_evaluate, batch_size)

        with open(log_file, "a") as f:
            f.write('{},{},{},'.format(acc_clean, acc_pgd, acc_df))

    def evaluate_model(self, model, batch_size=128):
        loss_value, acc = model.evaluate(self.x_evaluate, self.y_evaluate, batch_size)
        log_acc_and_loss(acc, loss_value, self.log_file)

        x_adv = self.pgd_util.pgd.generate_np(self.x_evaluate)
        loss_value, acc = model.evaluate(x_adv, self.y_evaluate, batch_size)
        log_acc_and_loss(acc, loss_value, self.log_file, "PGD")
#        save_perturbed_data(x_adv, "PGD20kPGD_ADV.txt")

        x_adv = self.deep_fool_util.create_adversaries(self.x_evaluate, self.y_evaluate)
        loss_value, acc = model.evaluate(x_adv, self.y_evaluate, batch_size)
        log_acc_and_loss(acc, loss_value, self.deep_fool_util.log_file, "Deep Fool")
#        save_perturbed_data(numpy.array(x_adv[0]), "PGD20kDF_ADV.txt")

