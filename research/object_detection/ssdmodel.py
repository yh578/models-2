import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from scipy import misc
import ntpath
from PIL import Image
from object_detection.calculate_iou_final import imageDetection
import time
from object_detection.utils import label_map_util

def ssdmodel(TEST_IMAGE_PATHS, MODEL_NAME, max_boxes_to_draw=30, min_score_thresh=.5):

    def get_num_pixels(filepath):
        width, height = Image.open(open(filepath)).size
        return width * height


    #if tf.__version__ < '1.4.0':
    #  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # ## Object detection imports


    # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
    # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

    # What model to download.
    #MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt') # Computer path to list of items with labels
    print(PATH_TO_LABELS)
    NUM_CLASSES = 90

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()    # Declares empty detection graph to be used during session
    with detection_graph.as_default():  # Declares detection graph as default graph
      od_graph_def = tf.GraphDef()      # Declares Graph Def (created by ProtoBuf)
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: # Imports frozen model with file des
        serialized_graph = fid.read()   # Reads from .pb file as binary
        od_graph_def.ParseFromString(serialized_graph)  # Loads the file into the graph_def variable
        tf.import_graph_def(od_graph_def, name='')  # Returns a list of operation or tensor objects and imports to detection graph
        #for node in od_graph_def.node:
        #    print(node.op)

    # Save graph to FileWriter to view on TensorBoard
    summary_writer = tf.summary.FileWriter('/tensorflow/logdir', detection_graph) #


    # Load categories and labels
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS) # Returns a StringIntLabelMapProto
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True) # Creates a dictionary of possible categories
    #print(categories)
    category_index = label_map_util.create_category_index(categories) # Creates a dictionary of categories, but array index is id


    # Declare array of (img file name, box coordinates)
    detections = []

    with detection_graph.as_default():  # Apply detection_graph as default
        # Sessions actually execute computations within the graph
      with tf.Session(graph=detection_graph) as sess: # Object in which Operation Objects are executed

        # Definite input and output Tensors for detection_graph

        #INPUT
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')     # gets tensor from graph by name
        # Each box represents a part of the image where a particular object was detected.

        #OUTPUT
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        total_start_time = time.time()
        list_run_time = []
        for ind, image_path in enumerate(TEST_IMAGE_PATHS):

          print('Image no: ')
          print(ind)

          image_np = misc.imread(image_path, mode='RGB')
          file_name = ntpath.basename(image_path)
          height_pix, width_pix, _ = image_np.shape
          print(image_np.shape)
          detect_struct = imageDetection()
          detect_struct.path = image_path
          detect_struct.basename = file_name


          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)

          # Actual detection.
          start_time = time.time()
          (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
          elapsed_time = time.time() - start_time
          list_run_time.append(elapsed_time)

          sqboxes = np.squeeze(boxes)
          print(file_name)

          sqscores = np.squeeze(scores)
          sqclasses = np.squeeze(classes)

          for i in range(sqboxes.shape[0]): #min(max_boxes_to_draw, sqboxes.shape[0])
              #if sqscores[i] > min_score_thresh: #scores is None or

                  #denormalize
            #print(sqboxes[i])
            sqboxes[i][0] = sqboxes[i][0]*height_pix #ymin
            sqboxes[i][1] = sqboxes[i][1]*width_pix #xmin
            sqboxes[i][2] = sqboxes[i][2]*height_pix #ymax
            sqboxes[i][3] = sqboxes[i][3]*width_pix #xmax

            detect_struct.pred_bbox_array.append(sqboxes[i])
            detect_struct.scores_array.append(sqscores[i])
            detect_struct.classes_array.append(sqclasses[i])
            detect_struct.pred_time.append(elapsed_time)

          detections.append(detect_struct)

        total_elapsed_time = time.time() - total_start_time

        # find mean
        avg_det_time = np.average(list_run_time)
        # find std deviation
        std_det_time = np.std(list_run_time)



    return detections, total_elapsed_time, avg_det_time, std_det_time
# END FUNCTION
