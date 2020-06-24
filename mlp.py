import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# --- Utility functions.
# --- Invariant moments of image
def invariant_image_moments(img):
    x = np.arange(img.shape[0]).reshape((1, img.shape[0]))
    y = np.arange(img.shape[1]).reshape((img.shape[1], 1))

    m00 = np.sum(img)
    x_m = np.sum(np.dot(x, img)) / m00
    y_m = np.sum(np.dot(img, y)) / m00
    x = x - x_m
    y = y - y_m

    n11 = np.dot(x, np.dot(img, y)) / np.power(m00, 2.0)
    n20 = np.sum(np.dot(x ** 2, img)) / np.power(m00, 2.0)
    n02 = np.sum(np.dot(img, y ** 2)) / np.power(m00, 2.0)
    n30 = np.sum(np.dot(x ** 3, img)) / np.power(m00, 2.5)
    n03 = np.sum(np.dot(img, y ** 3)) / np.power(m00, 2.5)
    n21 = np.dot(x ** 2, np.dot(img, y)) / np.power(m00, 2.5)
    n12 = np.dot(x, np.dot(img, y ** 2)) / np.power(m00, 2.5)

    n = np.zeros(7)
    n[0] = n20 + n02
    n[1] = (n20 - n02) ** 2 + 4.0 * n11 ** 2
    n[2] = (n30 - 3.0 * n12) ** 2 + (n03 - 3.0 * n21) ** 2
    n[3] = (n30 + n12) ** 2 + (n03 + n21) ** 2
    n[4] = (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2) - (3.0 * n21 - n03) * (
            n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    n[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4.0 * n11 * (n30 + n12) * (n03 + n21)
    n[6] = (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2) - (n30 - 3.0 * n12) * (
            n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    return n


def prepare_datasets():
    # --- Gather input images
    categories = os.listdir("data")
    data = list()
    for num, cat in enumerate(categories):
        print(F"Loading category: {cat}")
        code = [0.0 for n in range(len(categories))]
        code[num] = 1.0
        images = list()
        filenames = os.listdir(os.path.join("data", cat))

        # Show examples of images:
        img_rgb = Image.open(os.path.join("data", cat, filenames[21]))
        plt.imshow(img_rgb)
        plt.title(F"Category: {cat}")
        plt.show()
        img_binary = Image.open(os.path.join("data", cat, filenames[21])).convert('1')
        plt.imshow(img_binary, cmap='gray')
        plt.title(F"Category: {cat} - Blackand white")
        plt.show()

        # --- Process all images
        for f in filenames:
            img = Image.open(os.path.join("data", cat, f)).convert('1')
            np_img = np.asarray(img)
            images.append(np_img)
        plt.show()
        p = Pool(8)
        res = p.map(invariant_image_moments, images)
        print(F"Gathered {len(filenames)} i mages of category {cat}, calculating moments")
        for m_num, moment in enumerate(res):
            print(F"\r{m_num + 1}/{len(filenames)}", end="")
            data.append((moment, code))
        p.close()
        p.join()
        print("")

    # --- Save parsed data for faster re-runs
    with open('cached_data.npy', 'wb') as f:
        np.save(f, data)

    with open('cached_data.npy', 'rb') as f:
        data = np.load(f, allow_pickle=True)

    # --- Change  order to random
    np.random.shuffle(data)

    # --- Train test split, 90%/10%
    split = int(9 * len(data) / 10)
    train_data, test_data = data[:split][:], data[split:]

    x_train = np.asarray([d[0] for d in train_data])
    y_train = np.asarray([d[1] for d in train_data]).astype(float)
    x_test = np.asarray([d[0] for d in test_data])
    y_test = np.asarray([d[1] for d in test_data]).astype(float)

    return x_train, y_train, x_test, y_test, categories


class Layer(object):
    last_in = None
    last_out = None

    def __init__(self, in_dim, out_dim, learning_rate):
        self.w = np.random.normal(0, 0.1, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.learning_rate = learning_rate

    @staticmethod
    def activation(t):
        return 1 / (1 + np.exp(-t))

    @staticmethod
    def activation_backward(x):
        return x * (1 - x)

    def forward(self, x):
        self.last_in = x
        self.last_out = self.activation(np.dot(x, self.w) + self.b)
        return self.last_out

    def backward(self, error):
        grad = error * self.activation_backward(self.last_out)
        dw = self.last_in.T.dot(grad)
        db = np.sum(error, axis=0, keepdims=True)
        self.w = self.w + dw * self.learning_rate
        self.b = self.b + db[0] * self.learning_rate
        err = grad.dot(self.w.T)
        return err


class Model(object):
    layer0 = None
    layer1 = None
    layer2 = None
    layer3 = None

    def __init__(self, in_dim, hidden1, hidden2, out_dim, learning_rate):
        self.layer1 = Layer(in_dim, hidden1, learning_rate)
        self.layer2 = Layer(hidden1, hidden2, learning_rate)
        self.layer3 = Layer(hidden2, out_dim, learning_rate)

    def forward(self, x):
        self.layer0 = x
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        return x

    def backward(self, e):
        e = self.layer3.backward(e)
        e = self.layer2.backward(e)
        self.layer1.backward(e)


def main(learning_rate=0.001):
    x_train, y_train, x_test, y_test, categories = prepare_datasets()

    errors = []
    val_errors = []
    model = Model(7, 14, 8, 4, learning_rate)

    for i in range(50):
        pred = model.forward(x_train)
        error = y_train - pred
        model.backward(error)

        errors.append(np.mean(np.power(np.abs(error), 2)))

        val_pred = model.forward(x_test)
        val_error = np.mean(np.power(np.abs(y_test - val_pred), 2))
        val_errors.append(val_error)

    print(F"Final error:{errors[-1]}, final accuracy: {(1 - errors[-1]) * 100}%")
    print(F"Final validation error:{val_errors[-1]}, final validation accuracy: {(1 - val_errors[-1]) * 100}%")
    plt.plot(errors)
    plt.xlabel('Training')
    plt.ylabel('Mean Square Error')
    plt.plot(val_errors)
    plt.show()


if __name__ == "__main__":
    main(learning_rate=0.001)
