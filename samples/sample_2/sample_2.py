###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012                   #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012                   #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

import _init_paths
import os
import matplotlib.pyplot as plt
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *


def get_gtboxes(folder_file_name):
    folder_path = f'D:\Work_file\Object-Detection-Metrics-3D\samples\sample_2\{folder_file_name}' # 指定文件夹路径
    result = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                for line in f:
                    data = line.strip().split()
                    x1, y1, z1, x2, y2, z2 = map(float, data[1:7])
                    result.append([x1, y1, z1, x2, y2, z2])
    return result


def get_predboxes(folder_file_name, confidence):
    folder_path = f'D:\Work_file\Object-Detection-Metrics-3D\samples\sample_2\{folder_file_name}' # 指定文件夹路径
    result = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                for line in f:
                    data = line.strip().split()
                    confi = float(data[1])
                    if confi >= confidence:
                        x1, y1, z1, x2, y2, z2 = map(float, data[2:8])
                        result.append([x1, y1, z1, x2, y2, z2])
    return result


def filter_boxes(gt, pred):
    result = []
    # print(f'the len(gt) is {gt}')
    # print(f'the len(pred) is {pred}')
    yes = 0
    no = 0
    for box in gt:
        overlap = False
        for p_box in pred:
            # print(f'p_box is {p_box}')
            # print(f"box is {box}")
            if (box[0] < p_box[3] and box[3] > p_box[0]) and (box[1] < p_box[4] and box[4] > p_box[1]) and (box[2] < p_box[5] and box[5] > p_box[2]):
                # print(f"yes -- {yes}")
                yes += 1
                # print(f'have overlap -- \nthe gtbox is {box}\nthe prebox is {p_box}')
                overlap = True
                break
        if not overlap:
            # print(f"no -- {no}")
            no += 1
            result.append(box)
    # print(result)
    return result


def getBoundingBoxes(confi, gt_filename, det_filename):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    # print(f'currentPath is {currentPath}')
    folderGT = os.path.join(currentPath, gt_filename)
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])
            y = float(splitLine[2])
            z = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            d = float(splitLine[6])
            iMagshape = splitLine[7]
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                z,
                w,
                h,
                d,
                CoordinatesType.Absolute, iMagshape,
                BBType.GroundTruth,
                format=BBFormat.XYZX2Y2Z2)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    folderDet = os.path.join(currentPath, det_filename)
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            if confidence < confi:
                continue
            else:
                x = float(splitLine[2])
                y = float(splitLine[3])
                z = float(splitLine[4])
                w = float(splitLine[5])
                h = float(splitLine[6])
                d = float(splitLine[7])
                iMagshape = splitLine[8]
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    z,
                    w,
                    h,
                    d,
                    CoordinatesType.Absolute, iMagshape,
                    BBType.Detected,
                    confidence,
                    format=BBFormat.XYZX2Y2Z2)
                allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes


def createImages(dictGroundTruth, dictDetected):
    """Create representative images with bounding boxes."""
    import numpy as np
    import cv2
    # Define image size
    width = 200
    height = 200
    # Loop through the dictionary with ground truth detections
    for key in dictGroundTruth:
        image = np.zeros((height, width, 3), np.uint8)
        gt_boundingboxes = dictGroundTruth[key]
        image = gt_boundingboxes.drawAllBoundingBoxes(image)
        detection_boundingboxes = dictDetected[key]
        image = detection_boundingboxes.drawAllBoundingBoxes(image)
        # Show detection and its GT
        cv2.imshow(key, image)
        cv2.waitKey()


def plot_froc(epoches, train_data):
    # 计算这个 fROC 曲线
    evaluator = Evaluator()
    false_positives_per_image = []
    true_positive_fractions = []
    for i in range(5, 105, 5):
        i = round(i * 0.01, 2)
        # 取出这个prediction boxes 和 gt boxes
        pred_bboxes = get_predboxes(folder_file_name=f'detections_{epoches}', confidence=i)
        if train_data == True:
            ground_truth_boxes = get_gtboxes(folder_file_name='train_groundtruths')
            boundingboxes = getBoundingBoxes(confi=i, gt_filename='train_groundtruths', det_filename=f'detections_{epoches}')
        else:
            ground_truth_boxes = get_gtboxes(folder_file_name='groundtruths')
            boundingboxes = getBoundingBoxes(confi=i, gt_filename='groundtruths', det_filename=f'detections_{epoches}')
        no_predbox = filter_boxes(ground_truth_boxes, pred_bboxes)
        FPs = (len(ground_truth_boxes) - len(no_predbox)) / len(ground_truth_boxes)
        # print(FPs)
        true_positive_fractions.append(FPs)
        metricsPerClass = evaluator.GetPascalVOCMetrics(
            boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=0.01,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
            # recall = pred_bboxes
        for mc in metricsPerClass:
            fp  = mc['total FP']
            if train_data == True:
                recall = fp / len(os.listdir('D:\Work_file\Object-Detection-Metrics-3D\samples\sample_2\\train_groundtruths'))
            else:
                recall = fp / len(os.listdir('D:\Work_file\Object-Detection-Metrics-3D\samples\sample_2\\groundtruths'))
            false_positives_per_image.append(recall)

    plt.plot(false_positives_per_image, true_positive_fractions, marker='o', label=epoches)
    # 为每个点添加标签
    labels = [f'{round(i * 0.01, 2)}' for i in range(5, 105, 5)]
    for i, label in enumerate(labels):
        plt.annotate(label, xy=(false_positives_per_image[i], true_positive_fractions[i]), xytext=(false_positives_per_image[i] + 0.006, true_positive_fractions[i] - 0.08),
                    arrowprops=dict(facecolor='gray', edgecolor='gray', arrowstyle='-'))


def plot_ap(txt_name, train_data, confi=0.15):
    # Read txt files containing bounding boxes (ground truth and detections)
    if train_data == True:
        boundingboxes = getBoundingBoxes(confi=confi, gt_filename='train_groundtruths', det_filename=txt_name)
    else:
        boundingboxes = getBoundingBoxes(confi=confi, gt_filename='groundtruths', det_filename=txt_name)
    # Uncomment the line below to generate images based on the bounding boxes
    # createImages(dictGroundTruth, dictDetected)
    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()
    ##############################################################
    # VOC PASCAL Metrics
    ##############################################################
    # Plot Precision x Recall curve
    evaluator.PlotPrecisionRecallCurve(
        boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.01,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=True)  # Plot the interpolated precision curve
    
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.01,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        average_precision = mc['AP']
        tp  = mc['total TP']
        fp   = mc['total FP']
        all_gt = mc['total positives']
        # Print AP per class
        print('%s: %f' % (c, average_precision))
        print(f'the all gt is {all_gt}')
        # print(f'the precision is {precision}')
        print(f'the tp is {tp}')
        print(f'the fp is {fp}\n')
        # print(f'the det_tp is {len(mc["det_tp"])}')
        # pred_bboxes = mc["det_tp"]
        # print(f'the iou_col is {mc["iou_col"]}')
        # print(f'the irec is {irec}')






if __name__ == '__main__':

    plot_ap(txt_name='detections_310', train_data=False, confi=0.15)
        
    plt.figure()
    plot_froc(epoches=310, train_data=False)
    plot_froc(epoches=360, train_data=False)
    # plt.plot(false_positives_per_image, true_positive_fractions, marker='o')
    plt.grid(True)
    plt.legend()
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    plt.xlabel('False Positives Per Image')
    plt.ylabel('True Positive Fraction (recall)')
    plt.title('fROC Curve')

    plt.show()
