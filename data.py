import os
import pandas as pd
video=[]
df=pd.read_csv('Assignment Data - Sheet1.csv')
urls=df['Video URL']
for i,url in enumerate(urls):
  print(url)
  os.system(f'wget {url} -O Video/video_{i}.mp4')
  video.append(f'Video/video_{i}.mp4')
df['video']=video

df.to_csv('updated_final.csv')