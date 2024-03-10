from process import *
import glob
from matplotlib import pyplot as plt
import os


def validate(
        etalon_path: str,
        img_path: str,
        savefig_path: str = None) -> None:

    etalon_img = cv2.imdecode(np.fromfile(etalon_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
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

    '''  Различные детекторы (не сработало) '''
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


    rows = 6
    columns = 4
    fig = plt.figure(figsize=(5*columns, rows*4))

    ''' Различные детекторы
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



    '''  Различные дескрипторы  
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
#     fig.add_subplot(rows, columns, 1)
#     plt.imshow(etalon_img, 'gray')
#     plt.title('Etalon')
#
#     fig.add_subplot(rows, columns, 2)
#     plt.imshow(etalon_cut_data['img_canny'], 'gray')
#     plt.title('Etalon Canny image')
#
#     fig.add_subplot(rows, columns, 3)
#     plt.imshow(etalon_cut_data['img_box'], 'gray')
#     plt.title('Etalon with box')
#
#     fig.add_subplot(rows, columns, 4)
#     plt.imshow(etalon_cut_data['img_cutted'], 'gray')
#     plt.title('Etalon cutted')


# # Проверяемое изображение
#     fig.add_subplot(rows, columns, 5)
#     plt.imshow(test_img, 'gray')
#     plt.title('Validation image')
#
#     fig.add_subplot(rows, columns, 6)
#     plt.imshow(test_cut_data['img_canny'], 'gray')
#     plt.title('Validation image with box')
#
#     fig.add_subplot(rows, columns, 7)
#     plt.imshow(test_cut_data['img_box'], 'gray')
#     plt.title('Validation image with box')
#
#     fig.add_subplot(rows, columns, 8)
#     plt.imshow(test_cut_data['img_cutted'], 'gray')
#     plt.title('Validation Image cutted')


# # differences1_data -- абсолютная разница между исходными изображениями
#     fig.add_subplot(rows, columns, 9)
#     plt.imshow(differences1_data['img_diff'])
#     plt.title('Images differences ABS')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 10)
#     plt.imshow(differences1_data['img_diff_erode'])
#     plt.title('Erode ABS')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 11)
#     plt.imshow(differences1_data['img_diff_dilate'])
#     plt.title('Dilate ABS')
#     plt.colorbar()


# differences1_superimpose_data -- абсолютная разница между изображениями с помощью SIFT
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



# # differences2_data -- абсолютная разница между изображениями с применением БИНАРИЗАЦИИ и SSIM
#     fig.add_subplot(rows, columns, 17)
#     plt.imshow(255 - differences2_data['img_ssim'])
#     plt.title('Images differences SSIM')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 18)
#     plt.imshow(differences2_data['img_diff_erode'])
#     plt.title('Erode SSIM')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 19)
#     plt.imshow(differences2_data['img_diff_dilate'])
#     plt.title('Dilate SSIM')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 20)
#     plt.imshow(differences2_data['img_thresh'])
#     plt.title('Thresh')
#     plt.colorbar()


# # differences2_superimpose_data -- абсолютная разница между изображениями с помощью SIFT с применением БИНАРИЗАЦИИ и SSIM
#     fig.add_subplot(rows, columns, 21)
#     plt.imshow(255 - differences2_superimpose_data['img_ssim'])
#     plt.title('Images differences SSIM with SIFT')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 22)
#     plt.imshow(differences2_superimpose_data['img_diff_erode'])
#     plt.title('Erode SSIM with SIFT')
#     plt.colorbar()
#
#     fig.add_subplot(rows, columns, 23)
#     plt.imshow(differences2_superimpose_data['img_diff_dilate'])
#     plt.title('Dilate SSIM with SIFT')
#     plt.colorbar()



#     if savefig_path is not None:
#         if dir_path := os.path.dirname(savefig_path):
#             os.makedirs(dir_path, exist_ok=True)
#         plt.savefig(savefig_path, bbox_inches='tight')

    plt.show()

def _print_(chip_type: str) -> None:
    images = glob.glob(f'data/Xrays/{chip_type}/*.jpg') + glob.glob(f'data/Xrays/{chip_type}/*.jpeg')
    etalon_path = images[0]
    img_path = images[1]
    validate(etalon_path, img_path)

# for i in range(1,15):
#     _print_(f'{i}')
_print_('5')