import csv
from object_detection.ssdmodel import *
import cv2
from sklearn import metrics
import matplotlib.pyplot as plt


# search directory for image files
# inputs: image directory, pre-allocated array for images
# outputs: an array of image directory paths

# for one image, there will be the truth detections
class truthStruct:
    def __init__(self):
        self.path = ''
        self.basename = ''
        self.t_bbox_array = []
        self.num_of_detection = []
        self.isUsed = []
        self.scores_array = []
        self.pred_bbox_map = []
        self.iou_map = []
        self.pred_time = []


# for each image, there can be multiple bounding boxes associated with it
class imageDetection:
    def __init__(self):
        self.path = ''
        self.basename = ''
        self.isUsed = []
        self.pred_bbox_array = []
        self.gt_bbox_map = []
        self.iou_map = []
        self.scores_array = []
        self.classes_array = []
        self.all_truths = []
        self.pred_time = []


def imgpaths_from_dir(dir_name, ):
    mylist = []
    for root, dirs, files in os.walk(dir_name, topdown=True):
        for name in files:
            temp = name
            _, ext = os.path.splitext(temp)
            if ('.jpg' == ext) or ('.png' == ext):  # can make more efficient later
                pathToImage = os.path.join(root, name)
                mylist.append(pathToImage)
    return mylist
    #  print(os.path.join(root,name))


def write_boxes_to_CSV(all_entries, pred_csv):
    pred_boxes = []
    with open(pred_csv, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(len(all_entries)):
            name = os.path.splitext(all_entries[i][0])[0]  # gets base filename without extension
            line = name + "," + str(all_entries[i][1][0]) + "," + str(all_entries[i][1][1]) + "," + str(
                all_entries[i][1][2]) + "," + str(all_entries[i][1][3]) + "\n"
            csvwriter.writerow([line])


def write_final_to_CSV(list_of_imageDetections, list_of_truthDetections, output_dir, total_run_time, mean_detection_time, std_det_time):
    model_name = os.path.basename((output_dir)) + "\n"
    csvpath = output_dir + "/data.csv"
    csvpath_delimited = output_dir + "/data_delimited.csv"

    with open(csvpath, 'w', newline = '') as csvfile, open(csvpath_delimited, 'w', newline = '') as csvfile_delimited:

        csvwrit_delim = csv.writer(csvfile_delimited)
        file_delim = []

        csvwriter = csv.writer(csvfile)
        header = "Name, Detection Number, Ground Truth Coordinates, Predicted Coordinates, Detected (0/1), IOU, Score, Time for Detection (ms) \n"
        file = []
        file.append(header)

        all_times = []
        for i in range(len(list_of_truthDetections)):
            curr_truth = list_of_truthDetections[i]
            name = curr_truth.basename

            for j in range(len(curr_truth.t_bbox_array)):
                num_detect = curr_truth.num_of_detection[j]
                truth_bbox = curr_truth.t_bbox_array[j]

                if (curr_truth.isUsed[j] == 1):  # prediction mapped
                    pred_bbox = curr_truth.pred_bbox_map[j]
                    detection = 1  # change detection bit to 1
                    iou = curr_truth.iou_map[j]
                    time = curr_truth.pred_time[j] * 1000
                    score = curr_truth.scores_array[j]
                else:
                    pred_bbox = [0, 0, 0, 0]
                    detection = 0
                    iou = -1
                    time = 0
                    score = 0

                truth_bbox2 = truth_bbox
                truth_bbox = map(str, truth_bbox)
                truth_bbox2 = map(str, truth_bbox2)

                pred_bbox2 = pred_bbox
                pred_bbox = map(str, pred_bbox)
                pred_bbox2 = map(str, pred_bbox2)
                line = name + "," + str(num_detect) + "," + '"[' + str(",".join(truth_bbox)) + ']"' + "," + '"[' + str(",".join(pred_bbox)) + ']"'+ "," + str(detection) + "," + str(iou) + "," + str(score) + "," + str(time) + "\n"
                line_delimited = name + "," + str(num_detect) + "," + str(",".join(truth_bbox2)) + "," + str(",".join(pred_bbox2)) + "," + str(detection) + "," + str(iou) + "," + str(score) + "," + str(time) + "\n"
                file.append(line)
                file_delim.append(line_delimited)

        roc_auc = get_plot_prediction_metrics(list_of_imageDetections, list_of_truthDetections, output_dir)

        line1 = "AUC (Only Predictions Matched with Ground Truth: ," + str(roc_auc) + "\n"
        #line2 = "Mean Detection Time (Only Predictions Matched with Ground Truth): ," + str(mean_matched_time) + "\n"
        #line3 = "Standard Deviation for Detection Time [ms] (Only Predictions Matched with Ground Truth): ," + str(std_matched_time) + "\n"
        line4 = "Total Model Run Time [s] (All Predictions): ," + str(total_run_time)
        line5 = "Mean Detection Time [ms](All Predictions): ," + str(mean_detection_time*1000)
        line6 = "Standard Deviation for Detection Time [ms] (All Predictions): " + str(std_det_time*1000)
        model_overview = [model_name, line1, line4, line5, line6]

        for line in model_overview:
            csvwriter.writerow([line])
        for line in file:
            csvwriter.writerow([line])
        for line in file_delim:
            csvwrit_delim.writerow([line])

def get_plot_prediction_metrics(list_of_imageDetections, list_of_truthDetections,  outputdir):
    all_scores = []
    all_truth = []
    for i in range(len(list_of_imageDetections)):
        detection = list_of_imageDetections[i]
        all_scores = all_scores + detection.scores_array
        all_truth = all_truth + detection.isUsed

    for i in range(len(list_of_truthDetections)):
        for j in range(len(list_of_truthDetections[i].isUsed)):
            if list_of_truthDetections[i].isUsed[j] != 1:
                all_truth.append(1)
                all_scores.append(0)


    fpr, tpr, thresholds = metrics.roc_curve(all_truth, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    title = outputdir + "/roc_plt.png"
    plt.savefig(title, dpi=100)

    return roc_auc


def read_boxes_csv(filepath):
    list_of_truthDetections = []
    truthDetection = truthStruct()  # create a new one

    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for ind, row in enumerate(csvreader):
            if row:
                if (row[0] != truthDetection.basename):  # file is different from previous truthDetection
                    if (truthDetection.basename != ''):  # not initial object
                        list_of_truthDetections.append(truthDetection)  # add old truthDetection to list

                    truthDetection = truthStruct()  # create a new one
                    truthDetection.path = ''
                    truthDetection.basename = row[0]
                # else use same truthDetection as before and just append the bbox
                if len(row) == 7 and row[6] == '0':
                    truthDetection.num_of_detection.append(row[1]) #appends num of detection to array
                    truthDetection.t_bbox_array.append(row[2:6])  # append coordinates to bbox array

        if (truthDetection.basename != ''):  # not initial object
            list_of_truthDetections.append(truthDetection)  # add old truthDetection to list

    return list_of_truthDetections


# assumes in numerical order
def calculate_all_iou(list_of_imageDetections, list_of_truthDetections):
    i = 0
    j = 0

    while i < len(list_of_imageDetections) and j < len(list_of_truthDetections):
        pred_filename = os.path.splitext(list_of_imageDetections[i].basename)[0]
        gt_filename = list_of_truthDetections[j].basename

        if int(pred_filename) == int(gt_filename):  # if same file name, start finding optimized iou
            # for all predicted bboxes, find the optimized iou
            all_truths = list_of_truthDetections[j].t_bbox_array[:]
            all_truths_copy = []
            for item in all_truths:
                all_truths_copy.append(tuple(item))
            list_of_imageDetections[i].all_truths = all_truths_copy

            for p in range(len(list_of_imageDetections[i].pred_bbox_array)):
                # define variables
                curr_pred_bbox = list_of_imageDetections[i].pred_bbox_array[p]
                max_iou = 0
                max_index = -1
                for k in range(len(list_of_truthDetections[j].t_bbox_array)):  # iterate over array of bbox truth detections to find iou
                    curr_gt_bbox = list_of_truthDetections[j].t_bbox_array[k]  # get kth element of bbox_array
                    curr_isUsed = list_of_truthDetections[j].isUsed[k]

                    if (curr_isUsed != 1):
                        iou = bb_intersection_over_union(curr_pred_bbox, curr_gt_bbox)
                        #print("iou" + str(iou))
                        if (iou >= max_iou and iou <= 1 and iou >= 0.5):  # get new max IOU
                            max_iou = iou
                            max_index = k

                if max_index != -1:  # valid iou
                    best_gt_bbox = list_of_truthDetections[j].t_bbox_array[max_index]
                    cpy = []
                    for item in best_gt_bbox:
                        cpy.append(item)

                    # save mapping of list to image detection
                    list_of_imageDetections[i].gt_bbox_map.insert(p, cpy)
                    list_of_imageDetections[i].iou_map.insert(p, max_iou)
                    list_of_imageDetections[i].isUsed[p] = 1
                    # save important prediction values
                    time = list_of_imageDetections[i].pred_time[p]
                    score = list_of_imageDetections[i].scores_array[p]
                    # save mapping to truth detection
                    list_of_truthDetections[j].isUsed[max_index] = 1
                    list_of_truthDetections[j].pred_bbox_map[max_index] = curr_pred_bbox
                    list_of_truthDetections[j].iou_map[max_index] = max_iou
                    list_of_truthDetections[j].scores_array[max_index] = score
                    list_of_truthDetections[j].pred_time[max_index] = time
                else:
                    list_of_imageDetections[i].gt_bbox_map[p] = [-1, -1, -1, -1]

            j = j + 1
            i = i + 1

        elif (int(gt_filename) < int(pred_filename)):
            j = j + 1
        else:  # pred_filename < gt_filename - cannot iterate to find the ground truth image
            list_of_imageDetections[i].gt_bbox_map[0] = [-1, -1, -1, -1]
            list_of_imageDetections[i].iou_map[0] = -1
            i = i + 1
    # end while

    return list_of_imageDetections, list_of_truthDetections  # completed mapping for all images


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    xDiff = (xB - xA + 1)
    yDiff = (yB - yA + 1)

    if xDiff < 0 or yDiff < 0:
        iou = -1
    else:
        # compute the area of intersection rectangle
        interArea = (xDiff) * (yDiff)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def draw_iou_boxes(list_of_imageDetections, OUTPUT_DIR):
    all_gt = []
    for i, image_path in enumerate(list_of_imageDetections):

        # set current image detection
        curr_detection = list_of_imageDetections[i]  # gets imageDetection struct

        # set output path directory
        out_path = OUTPUT_DIR + "/" + curr_detection.basename

        # load the image
        image = cv2.imread(curr_detection.path)

        # draw truth
        for k in range(len(curr_detection.all_truths)):
            tr_bbox = curr_detection.all_truths[k]
            t_tl = tuple(list(map(int, tr_bbox[:2])))
            t_br = tuple(list(map(int, tr_bbox[2:])))
            cv2.rectangle(image, t_tl, t_br, (255, 0, 0), 2)  # all real

        for j in range(len(curr_detection.pred_bbox_array)):
            iou = curr_detection.iou_map[j]
            gt_bbox = curr_detection.gt_bbox_map[j]
            pred_bbox = curr_detection.pred_bbox_array[j]

            entry = [curr_detection.basename, gt_bbox]
            all_gt.append(entry)

            # draw predicted boxes
            p_tl = tuple(list(map(int, pred_bbox[:2])))
            p_br = tuple(list(map(int, pred_bbox[2:])))
            #cv2.rectangle(image, p_tl, p_br, (0, 255, 255), 2)  # pred = yellow

            if iou != -1:  # no valid iou
                textx = gt_bbox[0] - 10  # xmin
                texty = gt_bbox[1] - 10  # ymin
                textloc = tuple(list(map(int, [textx, texty])))

                gt_tl = tuple(list(map(int, gt_bbox[:2])))
                gt_br = tuple(list(map(int, gt_bbox[2:])))

                # BGR color parameters
                cv2.rectangle(image, gt_tl, gt_br, (0, 255, 0), 2)  # truth == GREEN

                p_tl = tuple(list(map(int, pred_bbox[:2])))
                p_br = tuple(list(map(int, pred_bbox[2:])))
                cv2.rectangle(image, p_tl, p_br, (0, 0, 150), 2)  # pred_coor == RED

                # compute the intersection over union and display it
                cv2.putText(image, "IoU: {:.4f}".format(iou), textloc,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print("{}: {:.4f}".format(image_path, iou))

        # show the output image

        cv2.imwrite(out_path, image)

    return all_gt
