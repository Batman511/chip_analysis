import sys
from process import *
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QScrollArea, QHBoxLayout, QGroupBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from pathlib import Path
import PyQt5
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins")

class ImageComparisonApp(QWidget):
    def __init__(self):
        super().__init__()

        self.image_folder = ""
        self.reference_image_path = ""
        self.result_images = []

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.folder_label = QLabel("Выберите папку с изображениями:")
        layout.addWidget(self.folder_label)

        self.folder_button = QPushButton("Выбрать папку")
        self.folder_button.clicked.connect(self.selectFolder)
        layout.addWidget(self.folder_button)

        self.reference_label = QLabel("Выберите эталон:")
        layout.addWidget(self.reference_label)

        self.reference_button = QPushButton("Выбрать эталон")
        self.reference_button.clicked.connect(self.selectReference)
        layout.addWidget(self.reference_button)

        # Виджет прокрутки
        self.scroll_area = QScrollArea()
        self.image_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_container)

        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

        self.setGeometry(100, 100, 1000, 800)
        self.setWindowTitle('Image Comparison App')

        # Поле вывода
        self.result_group_box = QGroupBox("Результат:")
        result_layout = QVBoxLayout()
        self.result_group_box.setLayout(result_layout)
        self.result_label = QLabel("Поле вывода будет здесь")
        result_layout.addWidget(self.result_label)
        self.result_group_box.hide()
        layout.addWidget(self.result_group_box)

    def selectFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder:
            self.image_folder = folder
            self.folder_label.setText(f"Выбрана папка: {folder}")

    def selectReference(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        reference_path, _ = QFileDialog.getOpenFileName(self, "Выберите эталон", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        if reference_path:
            self.reference_image_path = reference_path
            self.reference_label.setText(f"Выбран эталон: {os.path.basename(reference_path)}")

            # Отображаем изображение эталона с использованием QLabel
            pixmap = QPixmap(reference_path)
            self.reference_image_label = QLabel()
            self.reference_image_label.setPixmap(pixmap)
            # Устанавливаем размер QLabel на размер изображения
            self.reference_image_label.setFixedSize(pixmap.width(), pixmap.height())

            layout = self.layout()
            layout.addWidget(self.reference_image_label)

            # Делаем кнопку "Изменить положение рамки" доступной
            self.change_position_button = QPushButton("Изменить положение рамки вокруг эталонной картинки")
            self.change_position_button.clicked.connect(self.changePosition)
            layout.addWidget(self.change_position_button)

            # Скрываем поле вывода
            self.result_group_box.hide()

    def changePosition(self):
        # Ваш код по изменению положения рамки вокруг эталонной картинки
        # Этот код будет вызываться при нажатии кнопки "Изменить положение рамки"

        # Пока просто выводим сообщение, чтобы было видно, что кнопка работает
        print("Изменение положения рамки")

        # Показываем поле вывода
        self.result_group_box.show()

    def compareImages(self):
        if not self.image_folder or not self.reference_image_path:
            return

        # Ваш код по сравнению изображений, сохраняющий результаты в self.result_images
        self.result_images = self.performImageComparison()

        # Отображаем результаты в виджете прокрутки
        self.displayComparisonResults()

        # Показываем поле вывода
        self.result_group_box.show()

    def performImageComparison(self):
        # Здесь должен быть ваш код сравнения изображений
        # Просто возвращаем пустой список для примера
        return []

    def displayComparisonResults(self):
        # Очищаем предыдущие результаты
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.takeAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Выводим результаты
        for result_image in self.result_images:
            result_pixmap = self.imageToPixmap(result_image)
            result_label = QLabel()
            result_label.setPixmap(result_pixmap)
            result_label.setAlignment(Qt.AlignCenter)
            self.image_layout.addWidget(result_label)

    def imageToPixmap(self, image):
        if image is not None:
            # Уменьшаем изображение в 2 раза
            resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

            # Преобразование BGR изображения в RGB
            image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            # Получение параметров изображения
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width

            # Создание объекта QImage
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Преобразование QImage в QPixmap
            pixmap = QPixmap.fromImage(q_image)

            return pixmap
        else:
            print("Ошибка: Изображение не задано.")
            return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageComparisonApp()
    ex.show()
    sys.exit(app.exec_())
