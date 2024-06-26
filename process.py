import pandas as pd
import numpy as np
import json
import cv2
import math
import os
from tqdm import tqdm
import networkx as nx
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

current_checkpoint = 0
prev_checkpoint = 1
dest_checkpoint = 0
base_model = ResNet50(weights='imagenet', include_top=False)


def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def load_features(feature_folder):
    feature_vectors = []
    image_paths = []

    for filename in tqdm(os.listdir(feature_folder)):
        feature_path = os.path.join(feature_folder, filename)
        feature_vector = np.load(feature_path)
        feature_vectors.append(feature_vector)
        image_paths.append(feature_path.replace(".npy", "").split("/")[-1])

    return np.array(feature_vectors), image_paths


def find_most_similar_images(query_features, all_features, image_paths, top_k=5):
    similarities = cosine_similarity(
        query_features.reshape(1, -1), all_features)
    similar_indices = similarities.argsort()[0][-top_k:][::-1]
    similar_images = [image_paths[i] for i in similar_indices]
    return similar_images


def processFrame(frame):
    img_array = preprocess_image(frame)
    features = base_model.predict(img_array)
    features = features.flatten()
    return features


def getCurrentCheckpoint(features, all_features, image_paths):

    global current_checkpoint
    global prev_checkpoint
    df = pd.read_csv('./labels.csv')
    predicted_checkpoint = int(df.loc[df['file'].isin(img for img in find_most_similar_images(
        features, all_features, image_paths, top_k=10)), 'labels'].value_counts().sort_values(ascending=False).index[0])
    if current_checkpoint != predicted_checkpoint:
        prev_checkpoint = current_checkpoint
        current_checkpoint = predicted_checkpoint
        return


def getInstruction():
    checkpoint_df = pd.read_csv('./checkpoint.csv')
    graph_dic = json.load(open('./neighbourMatrix.json'))

    G = nx.DiGraph()

    for key, val in graph_dic.items():

        curr_check = int(key)
        curr_coordinates = checkpoint_df.loc[checkpoint_df['labels']
                                             == curr_check, 'coordinates'].values[0].split('_')
        for i in val:
            neighbour = int(i)
            neighbour_coord = checkpoint_df.loc[checkpoint_df['labels']
                                                == neighbour, 'coordinates'].values[0].split('_')

            distance = math.sqrt((int(curr_coordinates[0]) - int(neighbour_coord[0]))**2 + (
                int(curr_coordinates[1]) - int(neighbour_coord[1]))**2)
            G.add_edge(curr_check, neighbour, weight=distance)
    shortest_path = nx.dijkstra_path(
        G, source=current_checkpoint, target=dest_checkpoint, weight='weight')
    print(current_checkpoint, dest_checkpoint, shortest_path)
    if len(shortest_path) == 1:
        return "Arrived at the destination"

    prev_coord = checkpoint_df.loc[checkpoint_df['labels']
                                   == prev_checkpoint, 'coordinates'].values[0].split('_')
    curr_coord = checkpoint_df.loc[checkpoint_df['labels'] ==
                                   current_checkpoint, 'coordinates'].values[0].split('_')

    dirr_vector = np.array(
        [int(curr_coord[0]) - int(prev_coord[0]), int(curr_coord[1]) - int(prev_coord[1])])

    next_checkpoint = shortest_path[1]

    next_coord = checkpoint_df.loc[checkpoint_df['labels']
                                   == next_checkpoint]['coordinates'].values[0].split('_')

    final_dirr_vector = np.array(
        [int(next_coord[0]) - int(curr_coord[0]), int(next_coord[1]) - int(curr_coord[1])])

    turn_angle = 0
    if np.linalg.norm(final_dirr_vector) != 0 and np.linalg.norm(dirr_vector) != 0:

        turn_angle = math.acos(final_dirr_vector.dot(dirr_vector) / (np.linalg.norm(
            final_dirr_vector) * np.linalg.norm(dirr_vector))) * 180 / math.pi

    if turn_angle < 45:
        return "Go straight"
    elif turn_angle > 45 and turn_angle < 135:
        return "Turn left"
    elif turn_angle > 135 and turn_angle < 225:
        return "Turn around and walk back"
    else:
        return "Turn right"


def set_dest_checkpoint(destination_checkpoint):
    global dest_checkpoint
    dest_checkpoint = destination_checkpoint


def setup(image):
    features = processFrame(image)
    all_features, image_paths = load_features('./features2')
    getCurrentCheckpoint(features, all_features, image_paths)
    return getInstruction()


if __name__ == '__main__':
    image_path = './images/Test_Image_1_550.jpg'
    image = cv2.imread(image_path)
    set_dest_checkpoint(8)
    print(setup(image))
