import mnist
import tensorflow as tf
import numpy as np
from pandas import DataFrame
from classes import *


def run_nn(train_images, train_labels, test_images, test_labels):
    FILTERS_AMOUNT = 8
    NODES_AMOUNT = 10  # оно же количество классов
    INITIAL_SIZE = 26  # const - размер изображения

    # 28x28x1 (исходные размеры) -> 26x26x8 (теряем по пикселю при прогоне матричных фильтров, получаем 8 изображений)
    conv = Conv3x3(FILTERS_AMOUNT)
    # 26x26x8 -> 13x13x8 (сжатие в 2 раза)
    pool = MaxPool2()
    # на вход соответственно получаем 13 * 13 * 8 пикселей
    softmax = Softmax((INITIAL_SIZE // 2) * (INITIAL_SIZE // 2) * FILTERS_AMOUNT, NODES_AMOUNT)

    # один шаг - применяем сверточный слой, пулинг, softmax
    def forward(image, label):
        # нормировка данных в изображении 0-255 => 0-1 => -0.5-0.5
        out = conv.forward((image / 255) - 0.5)
        out = pool.forward(out)
        # на выходе получаем доли
        out = softmax.forward(out)
        loss = -np.log(out[label])
        # 1 - угадано, 0 - нет
        acc = 1 if np.argmax(out) == label else 0
        return out, loss, acc

    def train(im, label):
        # запуск слоя из трех составляющих - предсказание сети
        # out - выходные вероятности, чем являлось входное изображение im
        out, loss, acc = forward(im, label)
        gradient = np.zeros(NODES_AMOUNT)
        # все нули, а угаданное получает вес. Далее распространяем ошибку назад (коррекция весов), чтобы при следующем запуске стало лучше
        gradient[label] = -1 / out[label]
        gradient = softmax.backprop(gradient)
        gradient = pool.backprop(gradient)
        gradient = conv.backprop(gradient)
        return loss, acc

    # начинаем обучение - 3 эпохи по 1000 изображений
    for epoch in range(3):
        print('--- Epoch %d ---' % (epoch + 1))
        # перемешиваем числа от 0 до n и соответственно в случайном порядке расставляем датасет
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            # вывод статистики каждые 100 шагов
            if i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0
            l, is_right = train(im, label)
            loss += l
            num_correct += is_right  # 1 or 0

    loss = 0
    num_correct = 0
    predictions = []

    # по массиву вероятностей все же получить предсказание (максимальная вероятность)
    def get_label(probs):
        label = 0
        label_prob = probs[0]
        for i in range(1, len(probs)):
            if probs[i] > label_prob:
                label = i
                label_prob = probs[i]
        return label

    # обучение закончено, установлены подходящие веса. Проверка на тестовом сете
    for im, label in zip(test_images, test_labels):
        out, l, is_right = forward(im, label)
        predictions.append(get_label(out))
        loss += l
        num_correct += is_right

    num_tests = len(test_images)
    print('test loss:', loss / num_tests)  # average loss
    print('test accuracy:', num_correct / num_tests)  # percent of right answers
    error_rate = 1 - num_correct / num_tests
    print('error rate:', error_rate)  # percent of wrong ones

    # матрица, что чем предсказано
    cm = [[0 for j in range(NODES_AMOUNT)] for i in range(NODES_AMOUNT)]
    for p, l in zip(predictions, test_labels):
        cm[l][p] += 1

    print(DataFrame(cm))

# --------------------------------------------------------------------
# работа сети на mnist, подбор параметров сети - размера матрицы, пулинг-сжатия, количества фильтров и тд.
run_nn(mnist.train_images()[:1000], mnist.train_labels()[:1000], mnist.test_images()[:1000], mnist.test_labels()[:1000])
# ---------------------------------------------------------------------
# работа сети на fminst - проверка подобранных параметров
# (train_fmnist_images, train_fmnist_labels), (test_fmnist_images, test_fmnist_labels) = tf.keras.datasets.fashion_mnist.load_data()
# run_nn(train_fmnist_images[:1000], train_fmnist_labels[:1000], test_fmnist_images[:1000], test_fmnist_labels[:1000])
