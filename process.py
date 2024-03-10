import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.spatial import distance as dist
from memoization import cached
from skimage.metrics import structural_similarity as compare_ssim
from typing import Dict

@cached
def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Return ordered coordinates.
    Args:
        pts (np.ndarray):
    Returns:
        np.ndarray: ordered coords.

    Example:
    order_points(np.array([[0,0],[0,1],[1,1],[1,0]]))
    array([[0., 0.],[1., 0.],[1., 1.],[0., 1.]], dtype=float32)
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

@cached
def get_window(
    img: np.ndarray,
    threshold_min: int = 100,
    threshold_max: int = 200,
    padding_inp: Dict[str,int] = None,
    padding_out: Dict[str,int] = None,
    dbscan_kwargs: Dict = None,
    max_points: int = 10000,
    ) -> dict:
    """
    Cropping the image on the chip.

    Args:
        img (np.ndarray): source image.
        threshold_min (int): first threshold for the hysteresis procedure.
        threshold_max (int): second  threshold for the hysteresis procedure.
        padding_inp (Dict[str,int]): padding on input image.
        - 'x' (int)
        - 'y' (int)
        padding_out (Dict[str,int]): padding on output image.
        - 'x' (int)
        - 'y' (int)
        dbscan_kwargs (Dict): kwargs for DBSCAN clustering.

    Returns:
        dict: dict with results
        - img_canny (np.ndarray): Image after applying the Canny algorithm for edge detection.
        - img_box (np.ndarray): Image with drawn rectangles.
        - img_cutted (np.ndarray): Cropped image.
        - points (np.ndarray): Coordinates of the vertices of the detected object.

    """
    if padding_inp is None:
        padding_inp = {
            'x': 2,
            'y': 2,
        }
    if padding_out is None:
        padding_out = {
            'x': 0,
            'y': 0,
        }
    if dbscan_kwargs is None:
        dbscan_kwargs = {
            'eps': 20,
            'min_samples': 3,
        }

    img_cutted_in = img[padding_inp['x']:-padding_inp['x'],padding_inp['y']:-padding_inp['y']]
    canny_img = cv2.Canny(img_cutted_in, threshold_min, threshold_max)
    contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    X = np.concatenate(contours).reshape([-1, 2])
    if X.shape[0] > max_points:
        X = X[np.random.choice(X.shape[0], max_points, replace=False)]
    clustering = DBSCAN(**dbscan_kwargs).fit(X)
    counter = Counter(clustering.labels_)
    class_id = counter.most_common()[0][0]
    merged_contours = np.concatenate(contours)[:, 0, :]
    filtered_contours = X[clustering.labels_ == class_id]
    new_contours = filtered_contours


    combined_contour = filtered_contours + np.array([padding_inp['y'], padding_inp['x']])
    rect = cv2.minAreaRect(combined_contour)

    rect = (rect[0], (rect[1][0] + padding_out['y'],
            rect[1][1] + padding_out['x']), rect[2])
    box = cv2.boxPoints(rect)
    box = order_points(box)
    box = np.int0(box)
    img_box = img.copy()
    cv2.drawContours(img_box, [box], 0, (0, 0, 255), 2)

    h, w = img.shape[:2]
    dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1],
                       [0, h-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)

    img_cutted = cv2.warpPerspective(img, matrix, (w, h))
    img_cutted = cv2.resize(img_cutted, ((
        int(np.linalg.norm(box[0]-box[1]))), int(np.linalg.norm(box[1]-box[2]))))

    return {
        'img_canny': canny_img,
        'img_box': img_box,
        'img_cutted': img_cutted,
        'points': box
    }

@cached
def update_image_by_etalon(img_etalon: np.ndarray, img: np.ndarray, flag_des='flann', flag_det='sift') -> dict:
    """
    Tries to superimpose one image onto another using SIFT.

    Args:
        img_etalon (np.ndarray): source etalon image.
        img (np.ndarray): source image.

    Returns:
        dict: dict with results
        - img_box (np.ndarray): Image with drawn rectangles.
        - img_matches (np.ndarray): Image with matches.
        - img_cutted (np.ndarray): Image with region of interest.
    """
    MIN_MATCH_COUNT = 5

    # Нахождение ключевых точек и их дескрипторов для обоих изображений
    if flag_det=='sift':
          sift = cv2.SIFT_create()
          kp1, des1 = sift.detectAndCompute(img_etalon,None)
          kp2, des2 = sift.detectAndCompute(img,None)

    if flag_det=='brisk':
        brisk = cv2.BRISK_create()
        kp1, des1 = brisk.detectAndCompute(img_etalon,None)
        kp2, des2 = brisk.detectAndCompute(img,None)




    # Инициализация и сопоставление дескрипторов
    if flag_des=='flann':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

    elif flag_des=='bf':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

    elif flag_des=='bf_norm_L2':
        bf = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1,des2,k=2)



    # Применение фильтра для удаления неправильных сопоставлений
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img_etalon.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img_box = img.copy()
        img_box = cv2.polylines(img_box,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA,)
        invM = cv2.getPerspectiveTransform(dst, pts)
        img_cuted = cv2.warpPerspective(img, invM, (w, h))
        img_cuted = cv2.resize(img_cuted, (w, h))
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        raise Exception('NotEnoughPoints')


    # Рисование сопоставлений на изображениях
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = matchesMask, # draw only inliers
        flags = 2)
    img_matches = cv2.drawMatches(img_etalon,kp1,img_box,kp2,good,None,**draw_params)

    return {
        'img_box': img_box,
        'img_matches': img_matches,
        'img_cutted': img_cuted,
    }


@cached
def find_differences1(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Compare two images and find the differences between them using various image processing techniques.

    Parameters:
        img1 (np.ndarray): The first input image for comparison.
        img2 (np.ndarray): The second input image for comparison.

    Returns:
        dict: dict with results
        - 'img_diff': The absolute pixel-wise difference between img1 and a resized version of img2.
        - 'img_diff_erode': The eroded version of the absolute difference image to remove small noise.
        - 'img_diff_dilate': The dilated version of the absolute difference image to emphasize differences.
    """
    temp_img2 = img2.copy()
    temp_img2 = cv2.resize(temp_img2, img1.shape[::-1])
    diff = cv2.absdiff(img1, temp_img2)

    # Вычисление корреляции
    correlation = cv2.matchTemplate(img1, temp_img2, cv2.TM_CCOEFF_NORMED)[0][0]

    kernel = np.ones((3, 3), np.uint8)
    diff_erode = cv2.erode(diff, kernel, iterations=2)
    diff_dilate = cv2.dilate(diff, kernel, iterations=2)

    return {
        'img_diff': diff,
        'img_diff_erode': diff_erode,
        'img_diff_dilate': diff_dilate,
        'correlation': correlation
    }

@cached
def find_differences2(img1: np.ndarray, img2: np.ndarray, TRESH=True):
    """
    Compare two input images and identify the differences between them using Structural
    Similarity Index (SSIM) and image thresholding.

    Parameters:
        - img1 (numpy.ndarray): The first input image for comparison.
        - img2 (numpy.ndarray): The second input image for comparison.

    Returns:
        dict: dict with results
        - "img_ssim" (numpy.ndarray): An image representing the visual differences between img1 and img2.
        - "img_thresh" (numpy.ndarray): A binary thresholded image highlighting the differences.
        - "img_diff_erode": The eroded version of the absolute difference image to remove small noise.
        - "img_diff_dilate": The dilated version of the absolute difference image to emphasize differences.
        - "score" (float): The SSIM score representing the structural similarity between img1 and img2.
    """
    (score, diff) = compare_ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    if TRESH:
        thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        tresh = diff

    kernel = np.ones((3, 3), np.uint8)
    diff_erode = cv2.erode(thresh, kernel, iterations=1)
    diff_dilate = cv2.dilate(thresh, kernel, iterations=1)
    return {
        "img_ssim": diff,
        "img_thresh": thresh,
        "img_diff_erode": diff_erode,
        "img_diff_dilate": diff_dilate,
        "score": score,
    }