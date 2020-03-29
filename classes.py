import numpy as np


# adam-алгоритм оптимизации. Параметры по умолчанию являются стандартными
class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.weights = weights

    # алгоритм коррекции весов в зависимости от градиента
    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.weights = self.weights - self.alpha * (m_hat / (np.sqrt(v_hat) - self.epsilon))
        return self.weights


# сверточный слой размера 3 на 3
class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # генерируется случайная матрица из num_filters 3х3
        self.filters = np.random.randn(num_filters, 3, 3) / 9
        self.adam = AdamOptimizer(self.filters)

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    # движение вперед (применить слой). Выход - матрица- пачка изображений для каждого фильтра с немного уменьшенным размером изображения
    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    # back propagation - обратное распространение ошибки. Коррекция параметров в зависимости от промахов
    def backprop(self, input_gradients):
        _filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                _filters[f] += input_gradients[i, j, f] * im_region
        self.filters = self.adam.backward_pass(_filters)
        return None


# pool-слой, сокращает размер изображений в блоке изображений  в 2 раза (берет максимум из кластера 2 на 2)
class MaxPool2:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backprop(self, input_gradients):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = input_gradients[i, j, f2]
                            # break

        return d_L_d_input


# normalized exponential function - получает пиксели на вход, примеряет к ним преобразование из весов, далее экспонента и взятие ее процентного вклада
class Softmax:
    def __init__(self, input_len, nodes):
        # матрица случайных весов input_len на nodes по nodes для каждого входящего пикселя
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)  # сдвиги (добавочные веса)
        self.adam_weights = AdamOptimizer(self.weights)
        self.adam_biases = AdamOptimizer(self.biases)

    # сами входные пиксели
    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, input_gradients):
        for i, gradient in enumerate(input_gradients):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)
            sum_exp = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (sum_exp ** 2)
            d_out_d_t[i] = t_exp[i] * (sum_exp - t_exp[i]) / (sum_exp ** 2)
            d_L_d_t = gradient * d_out_d_t
            d_L_d_w = self.last_input[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_inputs = self.weights @ d_L_d_t
            self.weights = self.adam_weights.backward_pass(d_L_d_w)
            self.biases = self.adam_biases.backward_pass(d_L_d_t)
            return d_L_d_inputs.reshape(self.last_input_shape)
