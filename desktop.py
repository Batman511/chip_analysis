import sys
import os
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGroupBox, QPushButton, QVBoxLayout, QFileDialog, QScrollArea, QHBoxLayout, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView
from PyQt5.QtCore import Qt

from process import *

# Была проблема в комфликте версий. Решение - указать на использование плагинов Qt, используемых PyQt5.
from pathlib import Path
import PyQt5
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins")


class ImageComparisonApp(QWidget):
    def __init__(self):
        super().__init__()

        self.folder_path = ""
        self.label_path = ""
        # self.reference_image_path = ""

        self.result_images = []
        # self.reference_image_label = QLabel()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Увеличим размер шрифта
        font = QFont()
        font.setPointSize(12)


        self.folder_label = QLabel("Выберите папку с изображениями:")
        self.folder_label.setFont(font)
        layout.addWidget(self.folder_label)

        self.folder_button = QPushButton("Выбрать папку")
        self.folder_button.setFont(font)
        self.folder_button.clicked.connect(self.selectFolder)
        layout.addWidget(self.folder_button)


        self.reference_label = QLabel("Выберите эталон:")
        self.reference_label.setFont(font)
        layout.addWidget(self.reference_label)

        self.reference_button = QPushButton("Выбрать эталон")
        self.reference_button.setFont(font)
        self.reference_button.clicked.connect(self.selectLabel)
        layout.addWidget(self.reference_button)

        # вывод выбранного эталона
        self.reference_image_label = QLabel()
        layout.addWidget(self.reference_image_label)


        self.change_position_button = QPushButton("Изменить положение рамки вокруг эталонной картинки")
        self.change_position_button.clicked.connect(self.changePosition)
        self.change_position_button.setEnabled(False)  # Кнопка недоступна до выбора эталона
        layout.addWidget(self.change_position_button)


        self.compare_button = QPushButton("Сравнить все изображения с эталоном")
        self.compare_button.setFont(font)
        self.compare_button.clicked.connect(self.compareImages)
        layout.addWidget(self.compare_button)

        self.images_layout = QHBoxLayout()
        layout.addLayout(self.images_layout)



        # Виджет прокрутки
        self.scroll_area = QScrollArea()
        self.image_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_container)


        # layout.addWidget(self.scroll_area)


        # self.result_label = QLabel("Результат:")
        # self.result_label.setFont(font)
        # layout.addWidget(self.result_label)
        #
        # self.result_view = QGraphicsView()
        # self.result_scene = QGraphicsScene()
        # self.result_view.setScene(self.result_scene)
        # layout.addWidget(self.result_view)

        self.setLayout(layout)


        self.setGeometry(500, 200, 700, 400)
        self.setWindowTitle('Image Comparison App')

        # Поле вывода
        # self.result_group_box = QGroupBox("Результат:")
        # result_layout = QVBoxLayout()
        # self.result_group_box.setLayout(result_layout)
        # self.result_label = QLabel("Поле вывода будет здесь")
        # result_layout.addWidget(self.result_label)
        # self.result_group_box.hide()
        # layout.addWidget(self.result_group_box)

    def selectFolder(self):
        """
        Обработчик нажатия кнопки для выбора папки
        """
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"Выбрана папка: {os.path.basename(folder)}")

    def selectLabel(self):
        """
        Обработчик нажатия кнопки для выбора и вывода эталона (вырезанная микросхема)
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        label_path, _ = QFileDialog.getOpenFileName(self, "Выберите эталон", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        if label_path:
            self.label_path = label_path
            self.reference_label.setText(f"Выбран эталон: {os.path.basename(os.path.splitext(label_path)[0])}")

            # Выводим изображение эталона
            etalon_img = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            etalon_cut_data = get_window(etalon_img)['img_cutted']

            pixmap = self.convertCvImageToPixmap(etalon_cut_data, 300,300)
            self.reference_image_label.setPixmap(pixmap)

    def convertCvImageToPixmap(self, cv_image, target_width=None, target_height=None):
        """
        Функция для преобразования cv-изображения в pixmap для вывода в окне приложения
        Задаем размер выходящего изображения (target_width, target_height)
        """

        if cv_image is not None:
            # Преобразование в RGB
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Если указаны желаемые размеры вывода, изменяем размер изображения
            if target_width is not None and target_height is not None:
                image_rgb = cv2.resize(image_rgb, (target_width, target_height))

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
    def compareImages(self):
        """
        Функция для сравнения изображений и вывода карты различий
        """

        """
        # if not self.image_folder or not self.label_path:
        #     return

        # Ваш код по сравнению изображений, сохраняющий результаты в self.result_images
        # find_differences1(etalon_cut_data['img_cutted'], superimpose_data['img_cutted'])['img_diff']
        # etalon_img = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # etalon_cut_data = get_window(etalon_img)['img_cutted']

        # pixmap = self.convertCvImageToPixmap(etalon_cut_data)
        # self.reference_image_label.setPixmap(pixmap)
        self.result_images = cv2.imdecode(np.fromfile(self.label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # Отображаем результаты в виджете прокрутки
        self.displayComparisonResults()

        # Показываем поле вывода
        self.result_group_box.show()
        """


        # Отображение первого изображения
        cv_image_1 = cv2.imdecode(np.fromfile(self.label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        cv_image = get_window(cv_image_1)['img_cutted']

        # Преобразуем изображение cv в QPixmap
        pixmap = self.convertCvImageToPixmap(cv_image, 200, 200)

        # Создаем три виджета QLabel для отображения изображений
        for _ in range(3):
            label = QLabel()
            label.setPixmap(pixmap)
            self.images_layout.addWidget(label)

        # Показываем место для изображений
        self.images_layout.parentWidget().show()


        # pixmap1 = QPixmap(self.label_path)
        # label1 = QLabel()
        # label1.setPixmap(pixmap1)
        # self.original_images_layout.addWidget(label1)
        #
        # # Отображение второго изображения
        # # Предположим, что путь ко второму изображению хранится в переменной self.reference_image_path
        # pixmap2 = QPixmap(self.reference_image_path)
        # label2 = QLabel()
        # label2.setPixmap(pixmap2)
        # self.original_images_layout.addWidget(label2)
        #
        # # Показываем поле вывода
        # self.original_images_layout.show()

    def changePosition(self):
        # Изменение положения рамки вокруг эталонной картинки
        # Этот код будет вызываться при нажатии кнопки "Изменить положение рамки"

        # Пока не реализовано
        self.result_group_box.show()


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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageComparisonApp()
    ex.show()
    sys.exit(app.exec_())