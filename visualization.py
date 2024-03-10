"""
get_window(img)  # возвращает 3 изображения и координаты рамки:
1) ['img_canny']   граница с Canny (предобработка)
2) ['img_box']    выделенная рамка микросхемы
3) ['img_cutted']   вырезанная микросхема
4) ['points']   координаты рамки


find_differences1(etalon, test)  # возвращает 3 изображения и корреляцию изображений:
1) ['img_diff']   абсолютная разница между исходными изображениями. Получается путем вычисления абсолютной разницы между значениями пикселей в двух изображениях.
2) ['img_diff_erode']   изображение абсолютных различий с применением операции Erode (эрозия). Удаляет шумы или области с низкими значениями различий (проходится круглым или квадратным ядром). Само ядро стоит подбирать (можно сделать настраеваевым параметром для пользователя). (3,3) работает лучше всего
3) ['img_diff_dilate']   изображение абсолютных различий с применением операции Dilated (дилатация). Применяется для  увеличения размера белых (присутствующих) областей в бинарном изображении, чтобы выделить различия и сделать их более заметными (полезно, если важными являются мелкие различия).
4) ['correlation'] корреляция между изображениями с помощью matchTemplate()


find_differences2(etalon, test)  # возвращает 4 изображения с применением БИНАРИЗАЦИИ и SSIM:
1) ['img_ssim']   карта SSIM (разница между изображениями). На карте SSIM можно обнаружить области, где изображения сильно различаются.
2) ['img_thresh']   граница с помощью метода threshhold + OTSU для выбора порога для изображения img_ssim
3) ['img_diff_erode']   изображение абсолютных различий с применением операции Erode (эрозия). Удаляет шумы или области с низкими значениями различий (проходится круглым или квадратным ядром). Само ядро стоит подбирать (можно сделать настраеваевым параметром для пользователя). (3,3) работает лучше всего
4) ['img_diff_dilate']  изображение абсолютных различий с применением операции Dilated (дилатация). Применяется для  увеличения размера белых (присутствующих) областей в бинарном изображении, чтобы выделить различия и сделать их более заметными (полезно, если важными являются мелкие различия).
5) ['score']    индекс структурного сходства (SSIM) между двумя изображениями. Получим число от -1 до 1, где 1 - идеальное сходство.



update_image_by_etalon(etalon,test)  # возвращает 3 изображения:
1) ['img_box']   выделенная рамка микросхемы
2) ['img_matches']   выделенные признаки изображения с помощью SIFT (поиск ключевых точек)
3) ['img_cutted']   дескриптор изображения. Дескриптор описывает окружающую область каждой ключевой точки. Это описание в дальнешем сравнивается. Используется гомографическая матрица




test.shape[::-1] - кортеж, представляющий размеры целевого изображения
cv2.resize(etalon, test.shape[::-1]) - возвращает изображение etalon с новыми размерами

sift.detectAndCompute(img,None) - возвращает координаты ключевых точек (keypoints) и их описание (descriptors)
FLANN_INDEX_KDTREE - алгоритм FLANN для сопоставления ключевых точек.
cv2.FlannBasedMatcher - результатом является список matches, содержащий для каждой точки из первого набора два ближайших соседа из второго набора.
cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) - использует два набора ключевых точек (первое и второе изображения). Возвращает гомографическую матрицу имаску, указывающую, какие сопоставления используются, а какие - нет.
cv2.drawMatches() - предназначенна для визуализации результатов сопоставления ключевых точек между двумя изображениями
"""

from process import *
import glob
from matplotlib import pyplot as plt
import os


def validate(
        etalon_path: str,
        img_path: str,
        savefig_path: str = None) -> None:

    etalon_img = cv2.imdecode(np.fromfile(etalon_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) # оригинал
    test_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    etalon_cut_data = get_window(etalon_img)
    test_cut_data = get_window(test_img)

    try:
        superimpose_data = update_image_by_etalon(
            etalon_cut_data['img_cutted'],
            test_cut_data['img_cutted']
            )
    except Exception as e:
        print('SIFT not working')
        print(e)
        return


    differences1_data = find_differences1(
        cv2.resize(etalon_cut_data['img_cutted'], test_cut_data['img_cutted'].shape[::-1]),
        test_cut_data['img_cutted'])

    differences2_data = find_differences2(
        cv2.resize(etalon_cut_data['img_cutted'], test_cut_data['img_cutted'].shape[::-1]),
        test_cut_data['img_cutted'])

    differences1_superimpose_data = find_differences1(etalon_cut_data['img_cutted'], superimpose_data['img_cutted'])
    differences2_superimpose_data = find_differences2(etalon_cut_data['img_cutted'], superimpose_data['img_cutted'])

    '''  Различные дескрипторы  '''
    '''
    # differences1_flann = find_differences1(etalon_cut_data['img_cutted'], update_image_by_etalon(
    #                                                                                                     etalon_cut_data['img_cutted'],
    #                                                                                                     test_cut_data['img_cutted'],
    #                                                                                                     'flann'
    #                                                                                                     )['img_cutted'])
    # differences1_bf = find_differences1(etalon_cut_data['img_cutted'], update_image_by_etalon(
    #                                                                                                     etalon_cut_data['img_cutted'],
    #                                                                                                     test_cut_data['img_cutted'],
    #                                                                                                     'bf'
    #                                                                                                     )['img_cutted'])
    # differences1_bf_norm_L2 = find_differences1(etalon_cut_data['img_cutted'], update_image_by_etalon(
    #                                                                                                     etalon_cut_data['img_cutted'],
    #                                                                                                     test_cut_data['img_cutted'],
    #                                                                                                     'bf_norm_L2'
    #                                                                                                     )['img_cutted'])
    '''

    '''  Различные детекторы не сработало '''
    '''
    # differences1_sift = find_differences1(etalon_cut_data['img_cutted'], update_image_by_etalon(
    #                                                                                                     etalon_cut_data['img_cutted'],
    #                                                                                                     test_cut_data['img_cutted'],
    #                                                                                                     flag_det = 'sift'
    #                                                                                                     )['img_cutted'])
    # differences1_surf = find_differences1(etalon_cut_data['img_cutted'], update_image_by_etalon(
    #                                                                                                     etalon_cut_data['img_cutted'],
    #                                                                                                     test_cut_data['img_cutted'],
    #                                                                                                     flag_det = 'brisk'
    #                                                                                                     )['img_cutted'])


    # differences1_bf_norm_L2 = find_differences1(etalon_cut_data['img_cutted'], update_image_by_etalon(
    #                                                                                                     etalon_cut_data['img_cutted'],
    #                                                                                                     test_cut_data['img_cutted'],
    #                                                                                                     'bf_norm_L2'
    #                                                                                                     )['img_cutted'])
    '''


    rows = 1
    columns = 4
    fig = plt.figure(figsize=(5*columns, rows*4))

    '''
    fig.add_subplot(rows, columns, 1)
    plt.imshow(differences1_flann['img_diff'])
    plt.title('Images differences ABS with SIFT and flann')
    plt.colorbar()

    fig.add_subplot(rows, columns, 2)
    plt.imshow(differences1_bf['img_diff'])
    plt.title('Images differences ABS with SIFT and bf')
    plt.colorbar()

    fig.add_subplot(rows, columns, 3)
    plt.imshow(differences1_bf_norm_L2['img_diff'])
    plt.title('Images differences ABS with SIFT and bf_norm_l2')
    plt.colorbar()
    '''



    '''  Различные дескрипторы  '''
    ''' 
    # differences2 = find_differences2(etalon_cut_data['img_cutted'], superimpose_data['img_cutted'])
    # differences2_norm = find_differences2(etalon_cut_data['img_cutted'], superimpose_data['img_cutted'], TRESH=False)
    # 
    # 
    # 
    # fig.add_subplot(rows, columns, 1)
    # plt.imshow(differences2['img_diff'])
    # plt.title('Images differences ABS with SIFT without norm')
    # plt.colorbar()
    # 
    # fig.add_subplot(rows, columns, 2)
    # plt.imshow(differences2_norm['img_diff'])
    # plt.title('Erode with SIFT')
    # plt.colorbar()
    '''



# # Эталон, выделение объекта
    fig.add_subplot(rows, columns, 1)
    plt.imshow(etalon_img, 'gray')
    plt.title('Etalon')

    fig.add_subplot(rows, columns, 2)
    plt.imshow(etalon_cut_data['img_canny'], 'gray')
    plt.title('Etalon Canny image')

    fig.add_subplot(rows, columns, 3)
    plt.imshow(etalon_cut_data['img_box'], 'gray')
    plt.title('Etalon with box')

    fig.add_subplot(rows, columns, 4)
    plt.imshow(etalon_cut_data['img_cutted'], 'gray')
    plt.title('Etalon cutted')
    plt.show()

# # Проверяемое изображение
#     fig.add_subplot(rows, columns, 5)
#     plt.imshow(test_img, 'gray')
#     plt.title('Validation image')

#     fig.add_subplot(rows, columns, 6)
#     plt.imshow(test_cut_data['img_canny'], 'gray')
#     plt.title('Validation image with box')

#     fig.add_subplot(rows, columns, 7)
#     plt.imshow(test_cut_data['img_box'], 'gray')
#     plt.title('Validation image with box')

#     fig.add_subplot(rows, columns, 8)
#     plt.imshow(test_cut_data['img_cutted'], 'gray')
#     plt.title('Validation Image cutted')


# # differences1_data
#     fig.add_subplot(rows, columns, 9)
#     plt.imshow(differences1_data['img_diff'])
#     plt.title('Images differences ABS')
#     plt.colorbar()

#     fig.add_subplot(rows, columns, 10)
#     plt.imshow(differences1_data['img_diff_erode'])
#     plt.title('Erode ABS')
#     plt.colorbar()

#     fig.add_subplot(rows, columns, 11)
#     plt.imshow(differences1_data['img_diff_dilate'])
#     plt.title('Dilate ABS')
#     plt.colorbar()


# differences1_superimpose_data
#     fig.add_subplot(rows, columns, 13)
#     plt.imshow(differences1_superimpose_data['img_diff'])
#     plt.title('Images differences ABS with SIFT')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 14)
#     plt.imshow(differences1_superimpose_data['img_diff_erode'])
#     plt.title('Erode with SIFT')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 15)
#     plt.imshow(differences1_superimpose_data['img_diff_dilate'])
#     plt.title('Dilate with SIFT')
#     plt.colorbar()



# # differences2_data
#     fig.add_subplot(rows, columns, 17)
#     plt.imshow(255 - differences2_data['img_ssim'])
#     plt.title('Images differences SSIM')
#     plt.colorbar()

#     fig.add_subplot(rows, columns, 18)
#     plt.imshow(differences2_data['img_diff_erode'])
#     plt.title('Erode SSIM')
#     plt.colorbar()

#     fig.add_subplot(rows, columns, 19)
#     plt.imshow(differences2_data['img_diff_dilate'])
#     plt.title('Dilate SSIM')
#     plt.colorbar()

#     fig.add_subplot(rows, columns, 20)
#     plt.imshow(differences2_data['img_thresh'])
#     plt.title('Thresh')
#     plt.colorbar()


# # differences2_superimpose_data
#     fig.add_subplot(rows, columns, 21)
#     plt.imshow(255 - differences2_superimpose_data['img_ssim'])
#     plt.title('Images differences SSIM with SIFT')
#     plt.colorbar()

#     fig.add_subplot(rows, columns, 22)
#     plt.imshow(differences2_superimpose_data['img_diff_erode'])
#     plt.title('Erode SSIM with SIFT')
#     plt.colorbar()

#     fig.add_subplot(rows, columns, 23)
#     plt.imshow(differences2_superimpose_data['img_diff_dilate'])
#     plt.title('Dilate SSIM with SIFT')
#     plt.colorbar()

#     k = """ пока не надо
#     fig.add_subplot(rows, columns, 24)
#     plt.imshow(superimpose_data['img_matches'])
#     plt.title('SIFT matches')
#     plt.colorbar()
# """

#     if savefig_path is not None:
#         if dir_path := os.path.dirname(savefig_path):
#             os.makedirs(dir_path, exist_ok=True)
#         plt.savefig(savefig_path, bbox_inches='tight')




def _print_(chip_type: str) -> None:
    images = glob.glob(f'data/Xrays/{chip_type}/*.jpg') + glob.glob(f'data/Xrays/{chip_type}/*.jpeg')
    etalon_path = images[0]
    img_path = images[1]
    validate(etalon_path, img_path)

# for i in range(1,15):
#     _print_(f'{i}')
_print_('5')