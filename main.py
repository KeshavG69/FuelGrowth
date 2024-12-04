import cv2
import os
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import glob
import torch
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.cluster import DBSCAN


detector = MTCNN()
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()



influencer_images_dir = "influencer_images"

os.makedirs(influencer_images_dir, exist_ok=True)


df=pd.read_csv('updated_final.csv')
video_ratings=df.rename(columns={'Performance': 'rating'})



CLUSTERING_THRESHOLD = 0.45
FRAME_SKIP = 25


face_embeddings = []
face_metadata = []
face_crops = []


def process_video(video_path):
    """Extract faces and embeddings from video frames."""
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_number % FRAME_SKIP != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            bounding_box = face['box']
            face_crop = extract_face(rgb_frame, bounding_box)
            if face_crop is not None:
                embedding = get_face_embedding(face_crop)
                face_embeddings.append(embedding)
                face_metadata.append({
                    "video": video_path,
                    "frame": frame_number
                })
                face_crops.append(face_crop)

    cap.release()

def extract_face(frame, box):
    """Extract a cropped face from the bounding box."""
    x, y, width, height = box
    x, y = max(0, x), max(0, y)
    face_crop = frame[y:y+height, x:x+width]
    return cv2.resize(face_crop, (160, 160))

def get_face_embedding(face_crop):
    """Get a face embedding using FaceNet."""
    face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float().unsqueeze(0)
    face_tensor = (face_tensor - 127.5) / 128.0  
    with torch.no_grad():
        embedding = facenet_model(face_tensor).numpy().flatten()
    return embedding

def cluster_faces(embeddings):
    """Cluster face embeddings to identify unique influencers."""
    dbscan = DBSCAN(eps=CLUSTERING_THRESHOLD, metric="cosine", min_samples=1)
    labels = dbscan.fit_predict(embeddings)
    return labels

def save_influencer_images(labels, crops):
    """Save a representative image for each cluster."""
    representative_images = {}
    for label, crop in zip(labels, crops):
        if label not in representative_images:
            image_path = os.path.join(influencer_images_dir, f"influencer_{label}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            representative_images[label] = image_path
    return representative_images


video_files = glob.glob("Video/*.mp4")

for video in video_files:
    process_video(video)


if face_embeddings:
    labels = cluster_faces(face_embeddings)
    face_metadata_df = pd.DataFrame(face_metadata)
    face_metadata_df['influencer_id'] = labels

    
    influencer_images = save_influencer_images(labels, face_crops)

    
    face_metadata_df['influencer_image'] = face_metadata_df['influencer_id'].apply(lambda x: influencer_images.get(x))

    
    merged_data = pd.merge(face_metadata_df, video_ratings, on='video', how='inner')   # Use inner join to ensure all videos have ratings

    
    if merged_data.isnull().any().any():
        raise ValueError("Unexpected NaN values found after merging data. Check video names and metadata.")

    
    average_ratings = merged_data.groupby('influencer_id').agg(
        average_rating=('rating', 'mean'),
        influencer_image=('influencer_image', 'first')  
    ).reset_index()

    
    if average_ratings['average_rating'].isnull().any():
        raise ValueError("NaN found in average ratings despite all videos having ratings.")

    


wb = Workbook()
ws = wb.active
ws.title = "Influencer Data"
cell_size = 200  
ws.column_dimensions['C'].width = cell_size / 7.5 
for i in range(2, len(df) + 2):  
    ws.row_dimensions[i].height = cell_size

average_ratings_sorted = average_ratings.sort_values(by='average_rating', ascending=False)
for row in dataframe_to_rows(average_ratings_sorted[['influencer_id', 'average_rating']], index=False, header=True):
    ws.append(row)
for idx, image_path in enumerate(average_ratings_sorted['influencer_image'], start=2): 
    try:
        img = Image(image_path)  
        img.height = cell_size  
        img.width = cell_size   
        ws.add_image(img, f'C{idx}') 
    except FileNotFoundError:
        print(f"Image not found: {image_path}")

wb.save("influencer_data_with_images.xlsx")
