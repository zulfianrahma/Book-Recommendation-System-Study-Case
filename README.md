# Laporan Proyek Machine Learning : Book-Recommendation-System-Study-Case
Oleh Zulfian Rahmadiansyah

## Project Overview

Di dunia yang sudah serba cepat saat ini, dimana semakin melimpah informasi yang dapat diakses dalam waktu yang singkat. Namun, dengan melimpahnya informasi yang ada, hal tersebut membuat pengguna semakin sulit untuk mengakses informasi yang tepat dengan segera karena pengguna perlu melakukan proses seleksi atas banyaknya informasi yang dapat diperoleh. Sebagai solusi atas kesulitan tersebut, Sistem Rekomendasi (*Recommendation System*) muncul dan berkembang. Sistem rekomendasi adalah teknologi perangkat lunak yang dirancang untuk menyaring informasi atau mengusulkan produk atau layanan yang paling tepat berdasarkan preferensi pengguna. Saat ini, popularitas dari sistem rekomendasi telah meningkat pesat dan telah diterapkan secara luas di berbagai bidang seperti musik, film, berita, kesehatan, rekomendasi artikel, dan sebagainya [1].
Dua kategori data utama biasanya digunakan untuk mengimplementasikan suatu sistem rekomendasi. Kategori pertama terdiri dari informasi karakteristik yang menyediakan data terkait asosiasi produk, seperti kata kunci, kategorisasi, profil pengguna seperti usia, preferensi, dan lokasi. Kategori kedua mencakup informasi tentang interaksi pengguna dengan item atau produk, yang melibatkan data seperti peringkat (*rating*) pengguna dan peringkat produk [2]. 
Sistem rekomendasi dapat diklasifikasikan menjadi tiga kategori berdasarkan bagaimana rekomendasi itu dilakukan: Rekomendasi berbasis konten (*Content-based recommendation*), Rekomendasi berbasis kolaborasi (*Collaborative filtering*), dan pendekatan hibrid (*hybrid*). Di antara sistem rekomendasi tersebut, *collaborative filtering* adalah salah satu teknik yang paling banyak digunakan untuk merekomendasikan suatu produk. Produk tersebut direkomendasikan kepada pengguna tertentu berdasarkan peringkat pengguna lain dalam sistem. Di sisi lain, pendekatan berbasis konten (*content-based recommendation*) melakukan prediksi berdasarkan karakteristik produk yang berasal dari riwayat pengguna di masa lalu. Misalnya, pengguna yang menyukai film *Action* mungkin akan direkomendasikan film yang dikategorikan sebagai film *Action*. Sedangkan, pendekatan hibrid (*hybrid*) melibatkan penggabungan teknik penyaringan (*filtering*) berbasis konten dan kolaboratif dengan cara yang berbeda [3].

## Business Understanding

Sistem rekomendasi adalah alat canggih yang membuat penelusuran konten menjadi lebih mudah. Sistem ini menyediakan cara untuk mempersonalisasi konten bagi pengguna. Ia seperti filter pada saluran hidran sehingga air yang keluar tidak berlebihan. Dalam kasus informasi di internet, sistem rekomendasi menyaring konten yang muncul di layar agar sesuai dengan kebutuhan atau preferensi. Dengan sistem ini, kita bisa menelusuri internet secara efektif dan tidak akan kewalahan dengan berbagai informasi yang bertebaran. Organisasi atau pelaku bisnis juga dapat meningkatkan revenue atau pemasukan dari transaksi yang terjadi sebagai output dari sistem rekomendasi.
Beberapa tujuan dan fungsi sistem rekomendasi secara umum adalah sebagai berikut:
-	Meningkatkan jumlah item yang terjual
-	Menjual item yang beragam
-	Meningkatkan kepuasan pengguna
-	Pemahaman yang lebih baik tentang preferensi pengguna

### Problem Statements
-	Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap pemodelan sistem rekomendasi?
-	Bagaimana meningkatkan relevansi dan keakuratan model sistem rekomendasi yang dirancang?

### Goals
-	Mengetahui fitur yang paling berkorelasi dengan harga rumah.
-	Membuat model machine learning yang dapat memberikan rekomendasi yang relevan seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
-	Menggunakan sistem rekomendasi berbasis konten (*content-based recommendation*)
-	Menggunakan sistem rekomendasi berbasis kolaborasi (*collaborative recommendation*)

## Data Understanding
Data yang digunakan adalah data yang berkaitan dengan buku, dimana data tersebut memuat karakteristik buku (seperti judul buku, ISBN, nama penulis, dan nama penerbit), data peringkat (peringkat pengguna, ISBN, dan id pengguna), serta data pengguna (id, lokasi, dan usia pengguna). Data tersebut bernama **Book Recommendation Dataset** dan diperoleh melalui [Kaggle] (https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Terdapat 3 berkas csv dalam data tersebut, yaitu berkas *Books.csv* yang memuat data terkait karakteristik buku, berkas *Ratings.csv* yang memuat data terkait peringkat buku, dan *Users.csv* yang memuat data terkait pengguna.  

Fitur - Fitur pada **Book Recommendation Dataset** adalah sebagai berikut:

Tabel 1. Variabel pada berkas *Books.csv*
|     **Fitur**    |             **Deskripsi**             | **Tipe Data** |
|:-------------------:|:-------------------------------------:|:-------------:|
|         ISBN        |          Nomor ISBN dari buku         |     object    |
|      Book-Title     |               Judul buku              |     object    |
|     Book-Author     |              Penulis buku             |     object    |
| Year-Of-Publication |          Tahun publikasi buku         |     object    |
|      Publisher      |             Nama penerbit             |     object    |
|     Image-URL-S     | Tautan URL gambar cover buku dengan S |     object    |
|     Image-URL-M     | Tautan URL gambar cover buku dengan M |     object    |
|     Image-URL-L     | Tautan URL gambar cover buku dengan L |     object    |

Tabel 2. Fitur pada berkas *Ratings.csv*
| **Fitur** |     **Deskripsi**    | **Tipe Data** |
|:------------:|:--------------------:|:-------------:|
|    User-ID   |      ID pengguna     |    integer    |
|     ISBN     | Nomor ISBN dari buku |     object    |
|  Book-Rating |    Peringkat buku    |    integer    |

Tabel 3. Fitur pada berkas *Users.csv*
| **Fitur** |  **Deskripsi**  | **Tipe Data** |
|:------------:|:---------------:|:-------------:|
|    User-ID   |   ID pengguna   |    integer    |
|   Location   | Lokasi pengguna |     object    |
|      Age     |  Usia pengguna  |     float     |


## Data Preprocessing
Sebelum data diolah pada tahap *data preparation*, data melalui proses pra-pemrosesan yaitu penggabungan data pada tiap berkas dengan langkah teknis sebagai berikut:
1.	Mendefinisikan  variabel **all_book_rate** dengan variabel **rating** yang telah kita ketahui sebelumnya.
```
all_book_rate = ratings
all_book_rate
```
2.	Menggabungkan variabel **all_book_rate** dengan fitur pada berkas *Books.csv* berdasarkan **placeID**
```
# Menggabungkan all_book_rate dengan dataframe books berdasarkan placeID
all_book = pd.merge(all_book_rate, books[['ISBN',	'Book-Title', 'Book-Author']], on='ISBN', how='left')

# Print dataframe all_book
all_book
```

Membuat variabel **all_book** yang menampung fitur – fitur pada variabel **all_book_rate** dan variabel **books**. Fitur – fitur digabungkan dengan mencocokkan pada fitur **placeID**.
Tidak semua fitur pada variabel **books** diambil dimana fitur penerbit buku (*Book-Author*) tidak dimasukkan dengan alasan penyederhanaan model.

## Data Preparation
Untuk meningkatkan kualitas data yang akan digunakan pada tahap modelling, dilakukan proses data preparation atau penyiapan data. Proses penyiapan data yang dilakukan dapat dibagi menjadi beberapa komponen, yaitu:
1.	Menghilangkan **missing value**
  -	Pertama – tama mendeteksi kehadiran **missing value** dengan fungsi **isnull()**. Jika ditemukan **missing value** maka akan dihilangkan dengan fungsi **dropna()** sehingga dihasilkan dataset yang bersih (*clean*)

2.	Mengurutkan data dan menghilangkan duplikasi berdasarkan fitur **ISBN**
  -	Semua fitur yang ada diurutkan berdasarkan fitur **ISBN**.
  -	Data yang akan digunakan dalam proses pemodelan hanyalah data yang unik. Oleh karena itu, diperlukan proses penghapusan data yang duplikat dengan fungsi drop_duplicates().
3.	Konversi data menjadi **list**
  -	Mengkonversi fitur **ISBN, Book-Title, dan Book-Author** supaya bisa diproses menjadi **dictionary**

4.	Membuat **dictionary**
  -	Membuat dictionary untuk menentukan pasangan key-value pada data **book_id, book_name, dan book_author**

## Model Development
Dalam pengembangan studi kasus ini, dirancang dua jenis sistem rekomendasi, yaitu:
  -	Sistem rekomendasi berbasis konten (*content based recommendation*)
  -	Sistem rekomendasi berbasis kolaborasi (*collaborative recommendation*)
## Model Development dengan Content Based Recommendation
Pengembangan model dengan sistem rekomendasi berbasis konten yang dirancang menggunakan menggunakan fungsi **tfidfvectorizer()** dari library sklearn dan teknik **Cosine Similarity**.
**Term Frequency-Inverse Document Frequency** (TF-IDF) adalah vectorizer text yang mengubah teks menjadi vektor yang dapat digunakan. Fungsi tersebut menggabungkan dua konsep, yaitu *Term Frequency* (TF) dan *Document Frequency* (DF). *Term frequency* adalah jumlah kemunculan istilah tertentu dalam sebuah dokumen. *Term frequency* menunjukkan seberapa penting suatu istilah tertentu dalam sebuah dokumen. *Document frequency* adalah jumlah dokumen yang mengandung istilah tertentu. *Document frequency* menunjukkan seberapa umum istilah tersebut [4].
**Cosine Similarity** adalah metrik yang digunakan untuk mengukur kesamaan dua vektor. Secara khusus, ini mengukur kesamaan arah atau orientasi vektor dengan mengabaikan perbedaan besaran atau skalanya. Kedua vektor harus menjadi bagian dari ruang hasil kali dalam yang sama, artinya mereka harus menghasilkan skalar melalui perkalian hasil kali dalam. Kemiripan dua vektor diukur dengan kosinus sudut di antara keduanya [5].

![cosine-similarity-vectors original](https://github.com/zulfianrahma/Book-Recommendation-System-Study-Case/assets/97383651/8f34db0d-ca95-4893-93c0-fc29c390370d)
Gambar 1. **Cosine Similarity**

Tahapan yang dilakukan dalam proses pengembangan model antara lain:
1.	Menentukan jumlah sampel yang digunakan
  -	Dataset yang telah melalui proses *data preparation* berjumlah 270.150 baris data. Oleh sebab itu, untuk meringankan beban komputasi, hanya diambil sejumlah sampel data yang akan digunakan dalam proses pengembangan model.
2.	Implementasi fungsi **tfidfvectorizer()**
3.	Implementasi teknik **Cosine Similiarity**
4.	Mendapatkan rekomendasi
  -	Dalam proses mendapatkan rekomendasi, perlu dibuat sebuah fungsi baru yang digunakan untuk menjalankan proses pencarian rekomendasi. Fungsi tersebut akan dinamakan fungsi **book_recommendations**.
  -	Fungsi **book_reccomendation** yang dibuat mempunyai parameter fungsi sebagai berikut:
      o	Judul_buku : Judul buku (index kemiripan dataframe).
      o	Similarity_data : Dataframe mengenai similarity yang telah kita definisikan sebelumnya.
      o	Items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘book_title’ dan ‘book_author’.
      o	k : Banyak rekomendasi yang ingin diberikan.
```
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
```

## Model Development dengan Collaborative Recommendation
Pengembangan model berbasis kolaborasi (*collaborative recommendation*) disusun dengan beberapa tahapan yang berbeda dengan pengembangan model berbasis konten (*content based recommendation*). Oleh sebab itu, dilakukan kembali tahapan pengolahan data kembali mulai *data understanding* supaya dataset yang dimiliki dapat digunakan oleh model yang dilatih.
### Data Understanding
Tahapan **data understanding** terdiri dari beberapa langkah teknis, yaitu:
1.	Import *library* 
2.	Menentukan jumlah sampel data yang digunakan
  -	Dataset yang telah melalui proses *data preparation* berjumlah 270.150 baris data. Oleh sebab itu, untuk meringankan beban komputasi, hanya diambil sejumlah sampel data yang akan digunakan dalam proses pengembangan model.
### Data Preparation
Tahapan *data preparation* terdiri dari beberapa langkah teknis, yaitu:
1.	Penyandian (*encoded*) fitur **User-ID** dan **ISBN** menjadi indeks integer
  -	Penyandian dilakukan untuk melindungi privasi dan keamanan identitas dari pengguna
2.	Pemetaan fitur **User-ID** dan **ISBN** ke dalam dataframe
3.	Pembagian dataset menjadi data latih dan data uji
  -	Dataset dibagi menjadi data latih dan data uji dengan proporsi data 80:20
### Model Development
Tahapan *model development* terdiri dari beberapa langkah teknis, yaitu:
1.	Membuat class RecommenderNet dengan keras Model class.
```
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
```
2.	Melakukan proses *compile* dan pelatihan terhadap model
### Visualisasi Metrik
Pada tahapan ini, dilakukan visualisasi metrik menggunakan matplotlib.
![download](https://github.com/zulfianrahma/Book-Recommendation-System-Study-Case/assets/97383651/b4f1daf2-c66a-44c1-9d05-5370d95de64c)
Gambar 2. Metrik Evaluasi Sistem Rekomendasi Berbasi Kolaborasi
### Mendapatkan Rekomendasi Buku
Pada tahapan ini, dilakukan dua tahap utama sebelum mendapatkan rekomendasi buku.
1.	Membuat variabel book_not_read dengan menggunakan operator bitwise (~) pada variabel book_read_by_user.
-	Variabel tersebut bertujuan untuk membedakan antara buku yang sudah pernah dibaca dengan buku yang belum pernah dibaca pada dataset
2.	Melakukan rekomendasi buku menggunakan fungsi model.predict() dari library Keras

## Daftar Pustaka
1.	G. Jain, T. Mahara, and S. C. Sharma, “Performance Evaluation of Time-based Recommendation System in Collaborative Filtering Technique,” Procedia Comput Sci, vol. 218, pp. 1834–1844, Jan. 2023, doi: 10.1016/J.PROCS.2023.01.161.
2.	C. Udokwu, R. Zimmermann, F. Darbanian, T. Obinwanne, and P. Brandtner, “Design and Implementation of a Product Recommendation System with Association and Clustering Algorithms,” Procedia Comput Sci, vol. 219, pp. 512–520, Jan. 2023, doi: 10.1016/J.PROCS.2023.01.319.
3.	P. Kumar and R. S. Thakur, “Recommendation system techniques and related issues: a survey,” International Journal of Information Technology (Singapore), vol. 10, no. 4, pp. 495–501, Dec. 2018, doi: 10.1007/S41870-018-0138-8/FIGURES/3.
4.	“TF-IDF Simplified. A short introduction to TF-IDF… | by Luthfi Ramadhan | Towards Data Science.” https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530 (accessed Jul. 26, 2023).
5.	“Cosine Similarity – LearnDataSci.” https://www.learndatasci.com/glossary/cosine-similarity/ (accessed Jul. 26, 2023).
 

**---Ini adalah bagian akhir laporan---**
