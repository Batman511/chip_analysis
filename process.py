import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
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
    >>> order_points(np.array([[0,0],[0,1],[1,1],[1,0]]))
    >>> array([[0., 0.],[1., 0.],[1., 1.],[0., 1.]], dtype=float32)
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
        - points (np.ndarray): Coordinates of the vertices of the detected object object.
        
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
    # img_denoised = cv2.fastNlMeansDenoising(img_cutted_in,None,40,7,21)
    # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    canny_img = cv2.Canny(img_cutted_in, threshold_min, threshold_max)
    contours, hierarchy = cv2.findContours(
        canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    X = np.concatenate(contours).reshape([-1, 2])
    clustering = DBSCAN(**dbscan_kwargs).fit(X)
    counter = Counter(clustering.labels_)
    class_id = counter.most_common()[0][0]
    merged_contours = np.concatenate(contours)[:, 0, :]
    filtered_contours = merged_contours[clustering.labels_ == class_id]
    temp_contours = [set([(x[0], x[1]) for x in cnt[:, 0, :]])
                     for cnt in contours]
    mask = [False for x in range(len(contours))]
    for point in filtered_contours:
        for i2, cnt in enumerate(temp_contours):
            if tuple(point) in cnt:
                mask[i2] = True
    filtered_contours = [cnt for i, cnt in enumerate(contours) if mask[i]]
    new_contours = filtered_contours

    # for cnt in filtered_contours:
    #     rect = cv2.minAreaRect(cnt)
    #     square = int(rect[1][0]*rect[1][1])
    #     if square < img.shape[0] * img.shape[1] * 0.8:
    #         new_contours.append(cnt)

    combined_contour = np.concatenate(new_contours[:]) + np.array([padding_inp['y'], padding_inp['x']])
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
def update_image_by_etalon(img_etalon: np.ndarray, img: np.ndarray) -> dict:
    """
    Tries to superimpose one image onto another using an SIFT.

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
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_etalon,None)
    kp2, des2 = sift.detectAndCompute(img,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
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
        # print(dst)
        img_box = cv2.polylines(img_box,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        invM = cv2.getPerspectiveTransform(dst, pts)
        img_cuted = cv2.warpPerspective(img, invM, (w, h))
        img_cuted = cv2.resize(img_cuted, (w, h))
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        raise Exception('NotEnoughPoints')

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
        - 'absdiff': The absolute pixel-wise difference between img1 and a resized version of img2.
        - 'diff_erode': The eroded version of the absolute difference image to remove small noise.
        - 'diff_dilate': The dilated version of the absolute difference image to emphasize differences.
    """
    temp_img2 = img2.copy()
    temp_img2 = cv2.resize(temp_img2, img1.shape[::-1])
    diff = cv2.absdiff(img1, temp_img2)
    kernel = np.ones((4, 4), np.uint8) 
    diff_erode = cv2.erode(diff, kernel, iterations=2) 
    diff_dilate = cv2.dilate(diff, kernel, iterations=2) 

    return {
        'img_diff': diff,
        'img_diff_erode': diff_erode,
        'img_diff_dilate': diff_dilate,
    }

@cached
def find_differences2(img1: np.ndarray, img2: np.ndarray):
    """
    Compare two input images and identify the differences between them using Structural
    Similarity Index (SSIM) and image thresholding.

    Parameters:
        - img1 (numpy.ndarray): The first input image for comparison.
        - img2 (numpy.ndarray): The second input image for comparison.

    Returns:
        dict: dict with results
        - "img_diff" (numpy.ndarray): An image representing the visual differences between img1 and img2.
        - "img_thresh" (numpy.ndarray): A binary thresholded image highlighting the differences.
        - "img_erode" (numpy.ndarray): A further processed version of the thresholded image using erosion.
        - "score" (float): The SSIM score representing the structural similarity between img1 and img2.
    """
    (score, diff) = compare_ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    kernel = np.ones((4, 4), np.uint8) 
    diff_erode = cv2.erode(thresh, kernel, iterations=1) 
    diff_dilate = cv2.dilate(thresh, kernel, iterations=1) 
    return {
        "img_diff": diff,
        "img_thresh": thresh,
        "img_diff_erode": diff_erode,
        "img_diff_dilate": diff_dilate,
        "score": score,
    }