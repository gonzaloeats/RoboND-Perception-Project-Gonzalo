#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # statistical outlier filtering
    outlier_filter  = cloud.make_statistical_outlier_filter()

    outlier_filter.set_mean_k(20)
    
    x=0.1

    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()
    # Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = .01
    #set voxel size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()


    # PassThrough Filter z axis
    passthrough_z = cloud_filtered.make_passthrough_filter()
    filter_axis_z  = 'z'
    passthrough_z .set_filter_field_name(filter_axis_z)
    axis_min_z  = 0.6
    axis_max_z  = 1.3
    passthrough_z .set_filter_limits(axis_min_z, axis_max_z)
    cloud_filtered = passthrough_z .filter()

    # PassThrough Filter y axis
    passthrough_y = cloud_filtered.make_passthrough_filter()
    filter_axis_y = 'y'
    passthrough_y.set_filter_field_name(filter_axis_y)
    axis_min_y = -0.4
    axis_max_y = 0.4
    passthrough_y.set_filter_limits(axis_min_y, axis_max_y)
    cloud_filtered = passthrough_y.filter()

    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.02
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    # Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # Create cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(90)
    ec.set_MaxClusterSize(2700)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []
    detected_objects_labels = []
    detected_objects = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)


    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        histogram_bins = 128

        # Extract histogram features
        # complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, nbins=histogram_bins,using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals,nbins=histogram_bins)
        feature = np.concatenate((chists, nhists))
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels),
                                            detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)


    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass



def pr2_mover(object_list):

    #  Initialize variables
    yaml_output_list = []
    #  Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # Parse parameters into individual variables
    for p in object_list_param:
        object_name = p['name'] 
        object_group = p['group']



    ## Loop through the pick list
        for obj,  val in enumerate(object_list):
            if object_name != val.label:
                continue

            # set ros type and assign scene number
            ros_scene_num = Int32()
            test_num=1
            ros_scene_num.data = test_num



            # Assign the arm to be used for pick_place
            ros_arm_to_use = String()
            if object_group == "green":
                ros_arm_to_use.data = "right"
            else:
                ros_arm_to_use.data = "left"
           # Get centroid 
            centroid_array = np.mean(ros_to_pcl(val.cloud).to_array(), axis=0)[0:3]
            centroid = [np.asscalar(x) for x in centroid_array]

            # Create 'place_pose' for the object
            ros_object_name = String()
            ros_object_name.data = object_name
            ros_pick_pos = Pose()
            ros_pick_pos.position.x = centroid[0]
            ros_pick_pos.position.y = centroid[1]
            ros_pick_pos.position.z = centroid[2]

            # Drop box position
            box_pos = [0,0,0]
            for box_params in dropbox_param:
                if box_params['group'] == object_group:
                    box_pos = box_params['position']
                    break

            #except rospy.ServiceException, e:
            #    print "Service call failed: %s"%e
            ros_place_pos = Pose()
            ros_place_pos.position.x = box_pos[0]
            ros_place_pos.position.y = box_pos[1]
            ros_place_pos.position.z = box_pos[2]

            yaml_list = make_yaml_dict(ros_scene_num ,
                    ros_arm_to_use, ros_object_name,
                    ros_pick_pos,ros_place_pos ) 
                #print ("Response: ",resp.success)

        # Output your request parameters into output yaml file
            yaml_output_list.append(yaml_list)
            print('processed %s' % ros_object_name.data)


            # removes object from object_list to move to next object
            del object_list[obj]
            break
    # Output request parameters into an output yaml file
    send_to_yaml('output_%i.yaml' % test_num, yaml_output_list)



if __name__ == '__main__':

    #  ROS node initialization
    rospy.init_node('pr2', anonymous=True)
    #  Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2,
            pcl_callback, queue_size=1)
    #  Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2,
            queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2,
            queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2,
            queue_size=1)
    object_markers_pub  = rospy.Publisher("/object_markers", Marker,
            queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray,
            queue_size=1)
    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
