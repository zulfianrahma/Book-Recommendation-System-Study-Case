# -*- coding: utf-8 -*-
"""Proyek Kedua : Sistem Rekomendasi - Zulfian Rahmadiansyah.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L5nCt9-55sGSRregwieAtYEIzL2ozmTb

# Model Development dengan Content Based Filtering

## Data Understanding

*   Instalasi opendatasets untuk proses import data dari Kaggle
"""

!pip install opendatasets

"""

*   Import dataset dari Kaggle (diperlukan username dan key dari akun Kaggle selama proses import dataset)"""

import opendatasets as od

# import dataset
od.download('https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset')

"""

*   Membaca data-data dalam dataset dengan menggunakan fungsi pandas.read_csv"""

import pandas as pd

books = pd.read_csv('/content/book-recommendation-dataset/Books.csv')
ratings = pd.read_csv('/content/book-recommendation-dataset/Ratings.csv')
users = pd.read_csv('/content/book-recommendation-dataset/Users.csv')

print('Jumlah data buku: ', len(books.ISBN.unique()))
print('Jumlah penulis buku: ', len(books['Book-Author'].unique()))
print('Jumlah penerbit buku: ', len(books['Publisher'].unique()))

"""### Univariate Exploratory Data Analysis

Variabel - variabel yang ada pada Book Recommendation Dataset adalah:

* Users : Memuat data pengguna. Perlu diketahui bahwa id pengguna (User-ID) sudah dianonimkan dan dipetakan ke dalam integer. Data demografi disediakan (Location, Age) jika data tersebut tersedia. Jika tidak, informasi tersebut akan mempunyai nilai NULL.
* Books : Buku diidentifikasi dengan nomor ISBN. Nomor ISBN yang tidak sesuai telah dihilangkan dari dataset. Selain itu, beberapa informasi berkaitan dengan buku telah disediakan (Book-Title, Book-Author, Year-Of-Publication, Publisher) dan diperoleh dari Amazon Web Services. Dalam kasus banyak penulis, hanya penulis pertama yang ditampilkan. Tautan URL untuk gambar cover buku juga disediakan dan memiliki tiga jenis kategori (Image-URL-S, Image-URL-M, Image-URL-L) yaitu small, medium, dan large. Tautan tersebut diarahkan pada halaman Amazon Web.
* Rating: Memuat informasi terkait penilaian buku. Nilai (Book-Rating) yang diberikan bisa secara eksplisit, diekspresikan dalam skala 1-10, atau secara implisit, diekspresikan sebagai nilai 0.

*   Melihat informasi pada variabel books
"""

books.info()

"""

*   Melihat ada berapa banyak entri buku yang unik berdasarkan ISBN"""

print('Jumlah data buku: ', len(books.ISBN.unique()))

"""

*   Mengecek missing value pada variabel books dengan fungsi isnull()"""

books.isnull().sum()

"""

*   Melihat dataframe pada variabel books menggunakan fungsi head()
"""

books.head()

"""

*   Melihat informasi terkait entri penulis buku yang unik"""

print('Jumlah penulis buku: ', len(books['Book-Author'].unique()))
print('Nama - nama penulis buku: ', books['Book-Author'].unique())

"""

*   Melihat informasi terkait entri penerbit buku yang unik"""

print('Jumlah penerbit buku: ', len(books['Publisher'].unique()))
print('Nama - nama penerbit buku: ', books['Publisher'].unique())

"""

*   Melihat informasi pada variabel ratings
"""

ratings.info()

"""

*   Melihat ada berapa banyak entri penilaian buku yang unik berdasarkan User-ID"""

print('Jumlah data penilaian buku: ', len(ratings['User-ID'].unique()))

"""

*   Mengecek missing value pada variabel ratings dengan fungsi isnull()"""

ratings.isnull().sum()

"""

*   Melihat dataframe pada variabel ratings menggunakan fungsi head()"""

ratings.head()

"""

*   Menghitung jumlah rating, menggabungkannya berdasarkan ISBN, dan kemudian diurutkan mulai dari rating maksimal
"""

ratings.groupby('ISBN').sum().sort_values(by='Book-Rating', ascending=False)

"""

*   Melihat distribusi statistik pada variabel ratings"""

ratings.describe()

"""

*   Melihat informasi pada variabel users
"""

users.info()

"""

*   Melihat ada berapa banyak entri pengguna yang unik berdasarkan User-ID"""

print('Jumlah data profil pengguna: ', len(users['User-ID'].unique()))

"""

*   Mengecek missing value pada variabel users dengan fungsi isnull()"""

users.isnull().sum()

"""

*   Melihat dataframe pada variabel users menggunakan fungsi head()"""

users.head()

"""## Data Preprocessing

### Menggabungkan Data

*   Mendefinisikan variabel all_book_rate dengan variabel rating yang telah kita ketahui sebelumnya.
"""

all_book_rate = ratings
all_book_rate

"""

*   Menggabungkan all_book_rate dengan dataframe books berdasarkan placeID"""

# Menggabungkan all_book_rate dengan dataframe books berdasarkan placeID
all_book = pd.merge(all_book_rate, books[['ISBN',	'Book-Title', 'Book-Author']], on='ISBN', how='left')

# Print dataframe all_book
all_book

"""## Data Preparation

*   Mengecek missing value pada dataframe all_book
"""

all_book.isnull().sum()

"""

*   Membersihkan missing value dengan fungsi dropna()
"""

all_book_clean = all_book.dropna()
all_book_clean

"""

*   Mengecek kembali missing value pada variabel all_book_clean"""

all_book_clean.isnull().sum()

"""

*   Mengurutkan buku berdasarkan ISBN kemudian memasukkannya ke dalam variabel preparation"""

preparation = all_book_clean.sort_values('ISBN', ascending=True)
preparation

"""

*   Kita hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, kita perlu menghapus data yang duplikat dengan fungsi drop_duplicates().
* Membuang data duplikat berdasarkan ISBN.
"""

# Membuang data duplikat pada variabel preparation
preparation = preparation.drop_duplicates('ISBN')
preparation

"""

*   Melakukan konversi data series menjadi list menggunakan fungsi tolist() dari library numpy."""

# Mengonversi data series ‘ISBN’ menjadi dalam bentuk list
book_id = preparation['ISBN'].tolist()

# Mengonversi data series ‘Book-Title’ menjadi dalam bentuk list
book_title = preparation['Book-Title'].tolist()

# Mengonversi data series ‘Book-Author’ menjadi dalam bentuk list
book_author = preparation['Book-Author'].tolist()

print(len(book_id))
print(len(book_title))
print(len(book_author))

"""

*   membuat dictionary untuk menentukan pasangan key-value pada data book_id, book_name, dan book_author"""

book_new = pd.DataFrame({
    'id': book_id,
    'book_title': book_title,
    'book_author': book_author
})

book_new

"""## Model Development

### TF-IDF Vectorizer

*   Untuk meringankan beban komputasi, hanya diambil sejumlah data sampel (n) dari total dataset yang dimiliki
"""

# menentukan jumlah sampel data yang ingin diambil
sample = 5000

book_new_sample = book_new.sample(n=sample)

"""

*   Mengecek lagi data yang kita miliki dan assign dataframe dari tahap sebelumnya ke dalam variabel data"""

data = book_new_sample
data.sample(5)

"""

*   Menggunakan fungsi tfidfvectorizer() dari library sklearn."""

from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()

# Melakukan perhitungan idf pada data book_author
tf.fit(data['book_author'])

# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names_out()

"""

*   Melakukan fit dan transformasi ke dalam bentuk matriks."""

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(data['book_author'])

# Melihat ukuran matrix tfidf
tfidf_matrix.shape

"""

*   Menghasilkan vektor tf-idf dalam bentuk matriks menggunakan fungsi todense()."""

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

"""

*   Membuat dataframe untuk melihat tf-idf matrix
"""

# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan jenis masakan
# Baris diisi dengan nama resto

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=data.book_title
).sample(22, axis=1).sample(10, axis=0)

"""### Cosine Similarity

*   Menghitung derajat kesamaan (similarity degree) antar restoran dengan teknik cosine similarity
"""

from sklearn.metrics.pairwise import cosine_similarity

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

"""

*   Melihat matriks kesamaan setiap buku dengan menampilkan judul buku dalam 5 sampel kolom (axis = 1) dan 10 sampel baris (axis=0)."""

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa judul buku
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['book_title'], columns=data['book_title'])
print('Shape:', cosine_sim_df.shape)

# Melihat similarity matrix pada setiap buku
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""## Mendapatkan Rekomendasi

*   Membuat fungsi book_recommendations dengan beberapa parameter sebagai berikut:

  * Judul_buku : Judul buku (index kemiripan dataframe).
  * Similarity_data : Dataframe mengenai similarity yang telah kita definisikan sebelumnya.
  * Items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘book_title’ dan ‘book_author’.
  * k : Banyak rekomendasi yang ingin diberikan.
"""

def resto_recommendations(judul_buku, similarity_data=cosine_sim_df, items=data[['book_title', 'book_author']], k=5):
    """
    Rekomendasi Resto berdasarkan kemiripan dataframe

    Parameter:
    ---
    book_title : tipe data string (str)
                Judul buku (index kemiripan dataframe)
    similarity_data : tipe data pd.DataFrame (object)
                      Kesamaan dataframe, simetrik, dengan buku sebagai
                      indeks dan kolom
    items : tipe data pd.DataFrame (object)
            Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
    k : tipe data integer (int)
        Banyaknya jumlah rekomendasi yang diberikan
    ---


    Pada index ini, kita mengambil k dengan nilai similarity terbesar
    pada index matrix yang diberikan (i).
    """


    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,judul_buku].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop judul_buku agar nama resto yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(judul_buku, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

"""

*   Melihat informasi terkait buku "Come Death and High Water (George and Molly Palmer-Jones Mysteries)"
"""

data[data.book_title.eq('Come Death and High Water (George and Molly Palmer-Jones Mysteries)')]

"""

*   Menemukan rekomendasi buku yang mirip dengan "Come Death and High Water (George and Molly Palmer-Jones Mysteries)"
"""

resto_recommendations('Come Death and High Water (George and Molly Palmer-Jones Mysteries)')

"""# Model Development dengan Collaborative Filtering

## Data Understanding

*   Import library yang diperlukan
"""

# Import library
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

"""

*   Untuk meringankan beban komputasi, hanya diambil sejumlah data sampel (n) dari total dataset yang dimiliki
"""

# menentukan jumlah sampel data yang ingin diambil
sample = 5000

ratings_sample = ratings.sample(n=sample)

"""

*   Mengecek apakah terdapat missing values"""

ratings_sample.isnull().sum()

"""

*   Membaca dataset"""

df = ratings_sample
df

"""## Data Preparation

*   Melakukan persiapan data untuk menyandikan (encode) fitur ‘User-ID’ ke dalam indeks integer.
"""

# Mengubah User-ID menjadi list tanpa nilai yang sama
user_ids = df['User-ID'].unique().tolist()
print('list User-ID: ', user_ids)

# Melakukan encoding User-ID
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded User-ID : ', user_to_user_encoded)

# Melakukan proses encoding angka ke ke User-ID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke User-ID: ', user_encoded_to_user)

"""

*   Melakukan persiapan data untuk menyandikan (encode) fitur ‘ISBN’ ke dalam indeks integer."""

# Mengubah ISBN menjadi list tanpa nilai yang sama
ISBN = df['ISBN'].unique().tolist()
print('list ISBN: ', ISBN)

# Melakukan encoding ISBN
book_to_book_encoded = {x: i for i, x in enumerate(ISBN)}
print('encoded ISBN : ', book_to_book_encoded)

# Melakukan proses encoding angka ke ke ISBN
book_encoded_to_book = {i: x for i, x in enumerate(ISBN)}
print('encoded angka ke ISBN: ', book_encoded_to_book)

"""

*   Memetakan User-ID dan ISBN ke dataframe yang berkaitan."""

# Mapping User-ID ke dataframe user
df['user'] = df['User-ID'].map(user_to_user_encoded)

# Mapping ISBN ke dataframe book
df['book'] = df['ISBN'].map(book_to_book_encoded)

"""

*   Mengecek beberapa hal dalam data seperti jumlah user, jumlah buku, dan mengubah nilai rating menjadi float.
"""

# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah buku
num_book = len(book_encoded_to_book)
print(num_book)

# Mengubah Book-Rating menjadi nilai float
df['Book-Rating'] = df['Book-Rating'].values.astype(np.float32)

# Nilai minimum Book-Rating
min_rating = min(df['Book-Rating'])

# Nilai maksimal Book-Rating
max_rating = max(df['Book-Rating'])

print('Number of User: {}, Number of Book: {}, Min Book-Rating: {}, Max Book-Rating: {}'.format(
    num_users, num_book, min_rating, max_rating
))

"""*   Tahap persiapan telah selesai. Berikut adalah hal-hal yang telah kita lakukan pada tahap ini:

  * Memahami data rating yang kita miliki.
  * Menyandikan (encode) fitur ‘User-ID’ dan ‘ISBN’ ke dalam indeks integer.
  * Memetakan ‘User-ID’ dan ‘ISBN’ ke dataframe yang berkaitan.
  * Mengecek beberapa hal dalam data seperti jumlah user, jumlah buku, kemudian mengubah nilai rating menjadi float.

## Membagi Data untuk Training dan Validasi

*   Mengacak datanya agar distribusinya menjadi random
"""

# Mengacak dataset
df = df.sample(frac=1, random_state=42)
df

"""

*   Membuat variabel x untuk mencocokkan data user dan buku menjadi satu value"""

x = df[['user', 'book']].values

"""


*   Membuat variabel y untuk membuat rating dari hasil
"""

y = df['Book-Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

"""

*   Membagi menjadi 80% data train dan 20% data validasi"""

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

"""## Model Development

*   Membuat class RecommenderNet dengan keras Model class.
"""

class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_book, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_book = num_book
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.book_embedding = layers.Embedding( # layer embeddings book
        num_book,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.resto_bias = layers.Embedding(num_book, 1) # layer embedding book bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    book_vector = self.book_embedding(inputs[:, 1]) # memanggil layer embedding 3
    book_bias = self.resto_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_book = tf.tensordot(user_vector, book_vector, 2)

    x = dot_user_book + user_bias + book_bias

    return tf.nn.sigmoid(x) # activation sigmoid

"""

*   Melakukan proses compile terhadap model
"""

model = RecommenderNet(num_users, num_book, 50) # inisialisasi model

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""

*   Memulai training
"""

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 100,
    validation_data = (x_val, y_val)
)

"""## Visualisasi Metrik

*   Plot metrik evaluasi dengan matplotlib.
"""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""## Mendapatkan Rekomendasi Buku

*   Membuat variabel book_not_read dengan menggunakan operator bitwise (~) pada variabel book_read_by_user.
"""

book_df = book_new
df = pd.read_csv('/content/book-recommendation-dataset/Ratings.csv')

# Mengambil sample user
user_id = df['User-ID'].sample(1).iloc[0]
book_read_by_user = df[df['User-ID'] == user_id]

# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html
book_not_read = book_df[~book_df['id'].isin(book_read_by_user.ISBN.values)]['id']
book_not_read = list(
    set(book_not_read)
    .intersection(set(book_to_book_encoded.keys()))
)

book_not_read = [[book_to_book_encoded.get(x)] for x in book_not_read]
user_encoder = user_to_user_encoded.get(user_id)
user_book_array = np.hstack(
    ([[user_encoder]] * len(book_not_read), book_not_read)
)

"""

*   Memperoleh rekomendasi buku menggunakan fungsi model.predict() dari library Keras"""

ratings = model.predict(user_book_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_book_ids = [
    book_encoded_to_book.get(book_not_read[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Book with high ratings from user')
print('----' * 8)

top_book_user = (
    book_read_by_user.sort_values(
        by = 'Book-Rating',
        ascending=False
    )
    .head(5)
    .ISBN.values
)

book_df_rows = book_df[book_df['id'].isin(top_book_user)]
for row in book_df_rows.itertuples():
    print(row.book_title, ':', row.book_author)

print('----' * 8)
print('Top 10 book recommendation')
print('----' * 8)

recommended_book = book_df[book_df['id'].isin(recommended_book_ids)]
for row in recommended_book.itertuples():
    print(row.book_title, ':', row.book_author)
