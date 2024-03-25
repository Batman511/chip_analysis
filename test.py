from process import *
from matplotlib import pyplot as plt
import os
import glob

label_path = 'C:/Users/User/Documents/материалы ВИШ/АСЭ/chip_analysis/data/Xrays/5/14270175-x1(12).jpg'
if os.path.isfile(label_path):
    print(f"Изображение существует")
else:
    print(f"НЕ существует.")



etalon_img = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
etalon_cut_data = get_window(etalon_img)
etalon_cut_data = etalon_cut_data['img_cutted']
# etalon_cut_data = get_window(etalon_img)['img_cutted']

# Выводим изображение эталона с помощью plt
fig = plt.figure(figsize=(5, 4))
fig.add_subplot(1, 1, 1)
plt.imshow(etalon_cut_data, 'gray')
plt.title('Эталонное изображение')
plt.show()