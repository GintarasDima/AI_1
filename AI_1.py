# распазнование рукописных цифр с помощью ИИ
# программа написана на Colab
# все блоки разделены пустым коментарием

import tensorflow as tf
import tensorflow.keras

# библиотека для вывода изображений
import matplotlib.pyplot as plt
#% matplotlib inline

# -- Импорт для построения модели: --
# импорт слоев
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
# импорт модели
from tensorflow.keras.models import Sequential
# импорт оптимайзера
from tensorflow.keras.optimizers import Adam
# Импортируем набор данных MNIST
from tensorflow.keras.datasets import mnist
# Подключение утилит для to_categorical
from tensorflow.keras import utils
# работа с массивами
import numpy as np
#


# Загрузка из облака данных Mnist
(x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
#


# Вывод формы данных для обучения
x_train_org.shape
#


# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train_org[i], cmap=plt.cm.binary)

plt.show()
#


# вывод метки класса для 25-ти картинок
print(y_train_org[:25])
#


# изменение формы входных картинок с 28х28 на 784
# первая ось остается без изменений, остальные складывается в вектор
x_train = x_train_org.reshape(x_train_org.shape[0], -1)
x_test = x_test_org.reshape(x_test_org.shape[0], -1)

# проверка результата
print(f'Форма обучающих данных: {x_train_org.shape} -> {x_train.shape}')
print(f'Форма тестовых данных: {x_test_org.shape} -> {x_test.shape}')
#


# Нормализация входных картинок
# Преобразование x_train в тип float32 (числа с плавающей точкой) и нормализация
x_train = x_train.astype('float32') / 255.

# Преобразование x_test в тип float32 (числа с плавающей точкой) и нормализация
x_test = x_test.astype('float32') / 255.

# задание константы количества распозноваемых классов
CLASS_COUNT = 10

# Преобразование ответов в формат one_hot_encoding
y_train = utils.to_categorical(y_train_org, CLASS_COUNT)
y_test = utils.to_categorical(y_test_org, CLASS_COUNT)

# Вывод формы y_train
# 60 тысяч примеров, каждый длины 10 по числу классов
print(y_train.shape)
#


# Нормализация входных картинок
# Преобразование x_train в тип float32 (числа с плавающей точкой) и нормализация
x_train = x_train.astype('float32') / 255.

# Преобразование x_test в тип float32 (числа с плавающей точкой) и нормализация
x_test = x_test.astype('float32') / 255.

# задание константы количества распозноваемых классов
CLASS_COUNT = 10

# Преобразование ответов в формат one_hot_encoding
y_train = utils.to_categorical(y_train_org, CLASS_COUNT)
y_test = utils.to_categorical(y_test_org, CLASS_COUNT)

# Вывод формы y_train
# 60 тысяч примеров, каждый длины 10 по числу классов
print(y_train.shape)
#


# Вывод формы массива меток
print(y_train_org.shape)
#


# Вывод метки, соответствующей 25 элементам
print(y_train_org[:25])
#


# создание последовательной модели
model = Sequential()
# добавление полносвязного слоя на 800 нейронов с relu-активацией
model.add(Dense(800, input_dim=784, activation='relu'))
# добавление полносвязного слоя на 400 нейронов с relu-активацией
model.add(Dense(400, input_dim=784, activation='relu'))
# добавление полносвязного слоя на 100 нейронов с relu-активацией
model.add(Dense(100, input_dim=784, activation='relu'))
# Добавление полносвязного слоя с количеством нейронов по числу классов с softmax-активацией
model.add(Dense(CLASS_COUNT, activation='sigmoid'))

# Компиляция модели (создаем оптимизатор и функцию ошибки)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# выводим структуру модели
model.summary()
#


# наглядная схема нейронной сети
utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
#


model.fit(x_train,        # обучающая выборка, входные данные
          y_train,        # обучающая выборка, выходные данные
          batch_size=128, # кол-во примеров, которое обрабатывает нейронка перед одним изменением весов
          epochs=20,      # количество эпох, когда нейронка обучается на всех примерах выборки
          verbose=1)      # 0 - не визуализировать ход обучения, 1 - визуализировать
#


# запись и вызов модели
model.save_weights('model.h5')
model.load_weights('model.h5')
#


# лучшая модель
model.evaluate(x_test, y_test)
#


# Предсказываем результат для тестовой выборки
n = np.random.randint(x_test_org.shape[0])
pred = model.predict(x_test)
plt.imshow(x_test_org[n], cmap=plt.cm.binary)
plt.show()
#


# Выбор нужной картинки из тестовой выборки
x = x_test[n]

# Проверка формы данных
print(x.shape)
#


# Добавление одной оси в начале, чтобы нейронка могла распознать пример
# Массив из одного примера, так как нейронка принимает именно массивы примеров (батчи) для распознавания
x = np.expand_dims(x, axis=0)

# Проверка формы данных
print(x.shape)
#


# Распознавание примера
prediction = model.predict(x)
# Вывод результата - вектор из 10 чисел
print(prediction)
#


sum(prediction[0])
#


# Получение и вывод индекса самого большого элемента (это значение цифры, которую распознала сеть)
pred = np.argmax(prediction)
print(f'Распознана цифра: {pred}')
#


# Вывод правильного ответа для сравнения
print(y_test_org[n])
