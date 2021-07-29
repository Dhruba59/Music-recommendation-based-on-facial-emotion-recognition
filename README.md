Music Recommendation based on Facial Emotion Recognition
========================================================

**Mood Detection** model can detect face from any image and then it can predict the emotion from that face.
We can do it from both still images and videos.
After predicting the emotion from face our recommender system take the predicted emotion as input and generate recommendation by processing a Spotify dataset from a kaggle contest. We predicted the music mood from a model trained with **data_moods.csv**. The recommender system will generate top 40 songs to recommend for a spotify playlist.

Mood Detection
--------------
**Mood Detection** model will predict one of the emotion among 7 emotions listed below-
* Happy
* Sad
* Angry
* Disgust
* Surprise
* Neutral
* Fear

Music Mood Prediction
---------------------

Every songs in the main dataset in the **Datasets.7z** folder predicted to one of the mood among 4 moods listed below-
* Happy
* Sad
* Energetic
* Calm

By using a music mood classifier model we predicted each songs mood in our intermediate dataset **kaggleSpotifyMusicMood** in the Dataset.7z folder.

Music Recommendation
--------------------
Our main project file is **music_recommender.ipynb** file. This recommendation system is using content based filtering. We follow these steps to recommend music-
* Dataset Pre-processing
* Feature Engineering
* Connect to Spotify API
* Create Playlist Vector
* Generate Recommendation using cosine similarity

So according to this project, we will take an image of an user and predict emotion using **Emotion Detection** model. By prioritizing the songs from our main dataset **kaggleSpotifyMoodFinal.csv** with music mood comparing with different face emotion, this system will generate top 40 songs to recommend for a particular spotify playlist. 




Spotify Dataset link: https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks **(Spotify Dataset 1922-2021, ~600k Tracks)**                               
Dataset for mood Classifier link: https://github.com/cristobalvch/Spotify-Machine-Learning/tree/master/data **(data_moods.csv)**
