{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "fd3918bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import json\n",
    "import re \n",
    "import sys\n",
    "import itertools\n",
    "\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense\n",
    "from tensorflow.keras.wrappers.scikit_learn import KerasClassifier\n",
    "from tensorflow.keras import utils\n",
    "\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.model_selection import cross_val_score, KFold, train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score\n",
    "\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "import spotipy\n",
    "from spotipy.oauth2 import SpotifyClientCredentials\n",
    "from spotipy.oauth2 import SpotifyOAuth\n",
    "import spotipy.util as util\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2948c0ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option(\"max_rows\", None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1625cc05",
   "metadata": {},
   "outputs": [],
   "source": [
    "spotify_df = pd.read_csv('kaggleSPOTIFY.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "07b2ceaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "spotify_df.dropna(subset=['consolidates_genre_lists'], inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ba3a8715",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Unnamed: 0                  0\n",
       "valence                     0\n",
       "year                        0\n",
       "acousticness                0\n",
       "artists                     0\n",
       "danceability                0\n",
       "duration_ms                 0\n",
       "energy                      0\n",
       "explicit                    0\n",
       "id                          0\n",
       "instrumentalness            0\n",
       "key                         0\n",
       "liveness                    0\n",
       "loudness                    0\n",
       "mode                        0\n",
       "name                        0\n",
       "popularity                  0\n",
       "release_date                0\n",
       "speechiness                 0\n",
       "tempo                       0\n",
       "artists_upd_v1              0\n",
       "artists_upd_v2              0\n",
       "artists_upd                 0\n",
       "artists_song                0\n",
       "consolidates_genre_lists    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "spotify_df.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "93e55cc3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>valence</th>\n",
       "      <th>year</th>\n",
       "      <th>acousticness</th>\n",
       "      <th>artists</th>\n",
       "      <th>danceability</th>\n",
       "      <th>duration_ms</th>\n",
       "      <th>energy</th>\n",
       "      <th>explicit</th>\n",
       "      <th>id</th>\n",
       "      <th>instrumentalness</th>\n",
       "      <th>key</th>\n",
       "      <th>liveness</th>\n",
       "      <th>loudness</th>\n",
       "      <th>mode</th>\n",
       "      <th>name</th>\n",
       "      <th>popularity</th>\n",
       "      <th>release_date</th>\n",
       "      <th>speechiness</th>\n",
       "      <th>tempo</th>\n",
       "      <th>artists_upd_v1</th>\n",
       "      <th>artists_upd_v2</th>\n",
       "      <th>artists_upd</th>\n",
       "      <th>artists_song</th>\n",
       "      <th>consolidates_genre_lists</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.177</td>\n",
       "      <td>1989</td>\n",
       "      <td>0.568</td>\n",
       "      <td>['조정현']</td>\n",
       "      <td>0.447</td>\n",
       "      <td>237688</td>\n",
       "      <td>0.2150</td>\n",
       "      <td>0</td>\n",
       "      <td>2ghebdwe2pNXT4eL34T7pW</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>10</td>\n",
       "      <td>0.0649</td>\n",
       "      <td>-16.478</td>\n",
       "      <td>1</td>\n",
       "      <td>그아픔까지사랑한거야</td>\n",
       "      <td>31</td>\n",
       "      <td>1989-06-15</td>\n",
       "      <td>0.0272</td>\n",
       "      <td>71.979</td>\n",
       "      <td>['조정현']</td>\n",
       "      <td>[]</td>\n",
       "      <td>['조정현']</td>\n",
       "      <td>조정현그아픔까지사랑한거야</td>\n",
       "      <td>['classic_korean_pop']</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0.352</td>\n",
       "      <td>1992</td>\n",
       "      <td>0.381</td>\n",
       "      <td>['黑豹']</td>\n",
       "      <td>0.353</td>\n",
       "      <td>316160</td>\n",
       "      <td>0.6860</td>\n",
       "      <td>0</td>\n",
       "      <td>3KIuCzckjdeeVuswPo20mC</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0568</td>\n",
       "      <td>-9.103</td>\n",
       "      <td>1</td>\n",
       "      <td>DON'T BREAK MY HEART</td>\n",
       "      <td>35</td>\n",
       "      <td>1992-12-22</td>\n",
       "      <td>0.0395</td>\n",
       "      <td>200.341</td>\n",
       "      <td>['黑豹']</td>\n",
       "      <td>[]</td>\n",
       "      <td>['黑豹']</td>\n",
       "      <td>黑豹DON'T BREAK MY HEART</td>\n",
       "      <td>['chinese_indie', 'chinese_indie_rock']</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0.458</td>\n",
       "      <td>1963</td>\n",
       "      <td>0.987</td>\n",
       "      <td>['黃國隆']</td>\n",
       "      <td>0.241</td>\n",
       "      <td>193480</td>\n",
       "      <td>0.0437</td>\n",
       "      <td>0</td>\n",
       "      <td>4prhqrLXYMjHJ6vpRAlasx</td>\n",
       "      <td>0.000453</td>\n",
       "      <td>5</td>\n",
       "      <td>0.2650</td>\n",
       "      <td>-24.571</td>\n",
       "      <td>1</td>\n",
       "      <td>藝旦調</td>\n",
       "      <td>23</td>\n",
       "      <td>1963-05-28</td>\n",
       "      <td>0.0443</td>\n",
       "      <td>85.936</td>\n",
       "      <td>['黃國隆']</td>\n",
       "      <td>[]</td>\n",
       "      <td>['黃國隆']</td>\n",
       "      <td>黃國隆藝旦調</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0.796</td>\n",
       "      <td>1963</td>\n",
       "      <td>0.852</td>\n",
       "      <td>['黃國隆', '王秋玉']</td>\n",
       "      <td>0.711</td>\n",
       "      <td>145720</td>\n",
       "      <td>0.1110</td>\n",
       "      <td>0</td>\n",
       "      <td>5xFXTvnEe03SyvFpo6pEaE</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0695</td>\n",
       "      <td>-20.741</td>\n",
       "      <td>0</td>\n",
       "      <td>草螟弄雞公</td>\n",
       "      <td>23</td>\n",
       "      <td>1963-05-28</td>\n",
       "      <td>0.0697</td>\n",
       "      <td>124.273</td>\n",
       "      <td>['黃國隆', '王秋玉']</td>\n",
       "      <td>[]</td>\n",
       "      <td>['黃國隆', '王秋玉']</td>\n",
       "      <td>黃國隆草螟弄雞公</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0.704</td>\n",
       "      <td>1963</td>\n",
       "      <td>0.771</td>\n",
       "      <td>['黃國隆']</td>\n",
       "      <td>0.610</td>\n",
       "      <td>208760</td>\n",
       "      <td>0.1750</td>\n",
       "      <td>0</td>\n",
       "      <td>6Pqs2suXEqCGx7Lxg5dlrB</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7</td>\n",
       "      <td>0.0309</td>\n",
       "      <td>-20.232</td>\n",
       "      <td>1</td>\n",
       "      <td>思想起</td>\n",
       "      <td>23</td>\n",
       "      <td>1963-05-28</td>\n",
       "      <td>0.0419</td>\n",
       "      <td>124.662</td>\n",
       "      <td>['黃國隆']</td>\n",
       "      <td>[]</td>\n",
       "      <td>['黃國隆']</td>\n",
       "      <td>黃國隆思想起</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  valence  year  acousticness         artists  danceability  \\\n",
       "0           0    0.177  1989         0.568         ['조정현']         0.447   \n",
       "1           1    0.352  1992         0.381          ['黑豹']         0.353   \n",
       "2           2    0.458  1963         0.987         ['黃國隆']         0.241   \n",
       "3           3    0.796  1963         0.852  ['黃國隆', '王秋玉']         0.711   \n",
       "4           4    0.704  1963         0.771         ['黃國隆']         0.610   \n",
       "\n",
       "   duration_ms  energy  explicit                      id  instrumentalness  \\\n",
       "0       237688  0.2150         0  2ghebdwe2pNXT4eL34T7pW          0.000001   \n",
       "1       316160  0.6860         0  3KIuCzckjdeeVuswPo20mC          0.000000   \n",
       "2       193480  0.0437         0  4prhqrLXYMjHJ6vpRAlasx          0.000453   \n",
       "3       145720  0.1110         0  5xFXTvnEe03SyvFpo6pEaE          0.000000   \n",
       "4       208760  0.1750         0  6Pqs2suXEqCGx7Lxg5dlrB          0.000000   \n",
       "\n",
       "   key  liveness  loudness  mode                  name  popularity  \\\n",
       "0   10    0.0649   -16.478     1            그아픔까지사랑한거야          31   \n",
       "1   11    0.0568    -9.103     1  DON'T BREAK MY HEART          35   \n",
       "2    5    0.2650   -24.571     1                   藝旦調          23   \n",
       "3    2    0.0695   -20.741     0                 草螟弄雞公          23   \n",
       "4    7    0.0309   -20.232     1                   思想起          23   \n",
       "\n",
       "  release_date  speechiness    tempo  artists_upd_v1 artists_upd_v2  \\\n",
       "0   1989-06-15       0.0272   71.979         ['조정현']             []   \n",
       "1   1992-12-22       0.0395  200.341          ['黑豹']             []   \n",
       "2   1963-05-28       0.0443   85.936         ['黃國隆']             []   \n",
       "3   1963-05-28       0.0697  124.273  ['黃國隆', '王秋玉']             []   \n",
       "4   1963-05-28       0.0419  124.662         ['黃國隆']             []   \n",
       "\n",
       "      artists_upd            artists_song  \\\n",
       "0         ['조정현']           조정현그아픔까지사랑한거야   \n",
       "1          ['黑豹']  黑豹DON'T BREAK MY HEART   \n",
       "2         ['黃國隆']                  黃國隆藝旦調   \n",
       "3  ['黃國隆', '王秋玉']                黃國隆草螟弄雞公   \n",
       "4         ['黃國隆']                  黃國隆思想起   \n",
       "\n",
       "                  consolidates_genre_lists  \n",
       "0                   ['classic_korean_pop']  \n",
       "1  ['chinese_indie', 'chinese_indie_rock']  \n",
       "2                                       []  \n",
       "3                                       []  \n",
       "4                                       []  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "spotify_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "717e3a8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "mood_prep = spotify_df[['duration_ms', 'danceability', 'acousticness', 'energy', 'instrumentalness',\n",
    "       'liveness', 'valence', 'loudness', 'speechiness', 'tempo']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e805d2b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['duration_ms', 'danceability', 'acousticness', 'energy',\n",
       "       'instrumentalness', 'liveness', 'valence', 'loudness', 'speechiness',\n",
       "       'tempo'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "col_features = mood_prep.columns[:]\n",
    "col_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "956a83aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "mood_trans = MinMaxScaler().fit_transform(mood_prep[col_features])\n",
    "mood_trans_np = np.array(mood_prep[col_features])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "51e7e119",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 3.37093e+05,  4.26000e-01,  1.01000e-01,  6.46000e-01,\n",
       "        0.00000e+00,  3.58000e-01,  8.21000e-01, -1.10330e+01,\n",
       "        1.32000e-01,  2.05703e+02])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mood_trans_np[10011]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "949657c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('data_moods.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "123c68a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "cl_features = df.columns[6:-3]\n",
    "X= MinMaxScaler().fit_transform(df[cl_features])\n",
    "X2 = np.array(df[cl_features])\n",
    "Y = df['mood']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "9fa2cc1c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>mood</th>\n",
       "      <th>encode</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Calm</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Energetic</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Happy</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Sad</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        mood  encode\n",
       "5       Calm       0\n",
       "4  Energetic       1\n",
       "0      Happy       2\n",
       "1        Sad       3"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "encoder = LabelEncoder()\n",
    "encoder.fit(Y)\n",
    "encoded_y = encoder.transform(Y)\n",
    "\n",
    "\n",
    "dummy_y = utils.to_categorical(encoded_y)\n",
    "\n",
    "X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)\n",
    "\n",
    "target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)\n",
    "target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d8214135",
   "metadata": {},
   "outputs": [],
   "source": [
    "def base_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(8,input_dim=10,activation='relu'))\n",
    "    model.add(Dense(4,activation='softmax'))\n",
    "    model.compile(loss='categorical_crossentropy',optimizer='adam',\n",
    "                 metrics=['accuracy'])\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "68af0ae5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Configure the model\n",
    "estimator = KerasClassifier(build_fn=base_model,epochs=300,batch_size=200,verbose=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "e5171b9d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:5 out of the last 5 calls to <function Model.make_test_function.<locals>.test_function at 0x0000024B775F3160> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n",
      "WARNING:tensorflow:6 out of the last 6 calls to <function Model.make_test_function.<locals>.test_function at 0x0000024B6F5F0820> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n",
      "Baseline: 79.74% (3.85%)\n"
     ]
    }
   ],
   "source": [
    "#Evaluate the model using KFold cross validation\n",
    "kfold = KFold(n_splits=10,shuffle=True)\n",
    "results = cross_val_score(estimator,X,encoded_y,cv=kfold)\n",
    "print(\"Baseline: %.2f%% (%.2f%%)\" % (results.mean()*100,results.std()*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "0ee7ab16",
   "metadata": {},
   "outputs": [],
   "source": [
    "estimator.fit(X_train,Y_train)\n",
    "y_preds = estimator.predict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "15268b69",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "7fbf085b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('minmaxscaler', MinMaxScaler()),\n",
       "                ('keras',\n",
       "                 <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000024B732B80D0>)])"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Join the model and the scaler in a Pipeline\n",
    "pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=300, batch_size=200,verbose=0))])\n",
    "#Fit the Pipeline\n",
    "pip.fit(X2,encoded_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "57fc6356",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_mood(preds):\n",
    "    \n",
    "#     pipe = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=300, batch_size=200,verbose=0))])\n",
    "#     #Fit the Pipeline\n",
    "#     pipe.fit(X2,encoded_y)\n",
    "\n",
    "    preds_features = np.array(preds[:]).reshape(-1,1).T\n",
    "\n",
    "    #Predict the features of the song\n",
    "    results = pip.predict(preds_features)\n",
    "\n",
    "    mood = np.array(target['mood'][target['encode']==int(results)])\n",
    "\n",
    "    return str(mood[0])\n",
    "    #print(f\"{name_song} by {artist} is a {mood[0].upper()} song\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "a69a760a",
   "metadata": {},
   "outputs": [],
   "source": [
    "res = []\n",
    "\n",
    "for i in range(len(mood_trans_np)):\n",
    "  res.append(predict_mood(mood_trans_np[i]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "ac12763e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(156410, 25)"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "spotify_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "0d5d2d9a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "156410\n"
     ]
    }
   ],
   "source": [
    "print(len(res))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "3db86f15",
   "metadata": {},
   "outputs": [],
   "source": [
    "spotify_df['Mood'] = np.resize(res,len(spotify_df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "cfca6f08",
   "metadata": {},
   "outputs": [],
   "source": [
    "spotify_df.to_csv('kaggleMusicMoodFinal.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "089849a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "76262"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res.count(\"Sad\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "0f90f767",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "35535"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res.count(\"Happy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "31f7a804",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "31769"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res.count(\"Energetic\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "5e7a6616",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "12844"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res.count(\"Calm\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "2ce9d82c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(156410, 26)"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "spotify_df.shape"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
