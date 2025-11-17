# Dokumentasi Proyek: Fake News Detection

## 1. Ringkasan Proyek

Proyek ini bertujuan untuk membangun model *deep learning* yang mampu mengklasifikasikan sebuah artikel berita sebagai **"Berita Asli" (True)** atau **"Berita Palsu" (Fake)** berdasarkan konten teksnya.

Model ini menggunakan arsitektur **LSTM (Long Short-Term Memory)** yang diimplementasikan dengan TensorFlow/Keras. Setelah melalui proses pelatihan, model ini berhasil mencapai akurasi sekitar **97.9%** pada data uji, menunjukkan kemampuannya yang tinggi dalam membedakan kedua jenis berita.

## 2. Sumber Data

* **Sumber:** Dataset "Fake and Real News Dataset" dari Kaggle.
    * *Link: [True and Fake dataset](kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* **File:**
    * `Data/Fake.csv`: Berisi artikel berita palsu.
    * `Data/True.csv`: Berisi artikel berita asli.

## 3. Pengaturan Lingkungan

Proyek ini dikembangkan menggunakan Python dalam lingkungan virtual (`venv_tf`) dan Jupyter Notebook.

* **Lingkungan Virtual:** `venv_tf`
* **Library Utama:**
    * `pandas`
    * `numpy`
    * `re` (Regular Expressions)
    * `nltk` (Natural Language Toolkit)
    * `tensorflow` / `keras`
    * `sklearn` (Scikit-learn)
    * `matplotlib` & `seaborn`
* **Paket NLTK:** Diperlukan unduhan untuk `stopwords`, `wordnet`, `omw-1.4`, dan `punkt`.
* **Hardware:** Model ini dilatih menggunakan **CPU** (berdasarkan log notebook: `Num GPUs Available: 0`).

## 4. Alur Kerja Proyek (Logika Notebook)

Alur kerja proyek di `main.ipynb` dapat dibagi menjadi beberapa tahap utama:

### 4.1. Pemuatan & Eksplorasi Data

1.  **Load Data:** `Fake.csv` dan `True.csv` dibaca ke dalam DataFrame Pandas (`fake` dan `true`).
2.  **Labeling:** Kolom `label` ditambahkan.
    * `0` untuk berita `fake`.
    * `1` untuk berita `true`.
3.  **Gabung Data:** Kedua DataFrame digabung menjadi satu DataFrame utama bernama `News`.
4.  **Eksplorasi Awal:**
    * `News.info()` menunjukkan ada 44.898 total entri sebelum pembersihan.
    * `News.isnull().sum()` mengonfirmasi **tidak ada nilai null** dalam dataset.
    * Visualisasi `sns.countplot` menunjukkan distribusi kelas yang relatif seimbang (meskipun ada lebih banyak data "fake").

### 4.2. Pembersihan Data

1.  **Hapus Kolom:** Kolom `title`, `date`, dan `subject` dihapus karena fokus analisis adalah pada konten `text`.
2.  **Hapus Duplikat:**
    * Dataset diperiksa (`News.duplicated().sum()`) dan ditemukan **6.251** baris duplikat.
    * Duplikat tersebut dihapus (`News.drop_duplicates()`).
    * Dataset akhir memiliki **38.647** entri unik.

### 4.3. Pra-pemrosesan Teks

Sebuah fungsi kustom `process_text(text)` didefinisikan untuk membersihkan setiap artikel berita. Langkah-langkahnya adalah:

1.  Menghapus spasi putih berlebih (`\s+`).
2.  Menghapus semua karakter spesial (`\W`).
3.  Menghapus semua karakter tunggal (`\s+[a-zA-Z]\s+`).
4.  Menghapus semua karakter non-alfabet (`[^a-zA-Z\s]`).
5.  Mengubah teks menjadi *lowercase*.
6.  **Tokenisasi:** Memecah teks menjadi daftar kata-kata.
7.  **Lemmatization:** Mengubah kata ke bentuk dasarnya (misal: "running" -> "run") menggunakan `WordNetLemmatizer`.
8.  Menghapus *stopwords* (kata umum seperti "dan", "di", "the").
9.  Menghapus kata-kata yang terlalu pendek (kurang dari 4 karakter).
10. **Deduplikasi Kata:** Menggunakan `np.unique` untuk menyimpan hanya kata-kata unik dalam satu artikel (dengan mempertahankan urutan kemunculan pertama).

### 4.4. Persiapan Data untuk Model

1.  **Pemisahan Fitur & Target:** Data dipisah menjadi `x` (teks) dan `y` (label).
2.  **Train-Test Split:** Data dibagi menjadi 80% data latih (30.917 sampel) dan 20% data uji (7.730 sampel).
3.  **Tokenisasi Keras:**
    * `Tokenizer` dari Keras di-*"fit"* pada data latih (`x_train`) untuk membuat indeks kosakata.
    * Ukuran kosakata (vocabulary size) yang ditemukan adalah **91.536** kata unik.
4.  **Konversi ke Sekuens:** Teks diubah menjadi sekuens angka (`texts_to_sequences`).
5.  **Padding:** Setiap sekuens dipastikan memiliki panjang yang sama, yaitu **150 kata** (`pad_sequences` dengan `maxlen=150`). Jika lebih pendek, akan ditambahkan 0; jika lebih panjang, akan dipotong.
6.  **Encoding Label:** Label `y` (0 dan 1) diubah menjadi format *one-hot encoding* (misal: 0 -> `[1, 0]`, 1 -> `[0, 1]`) menggunakan `LabelEncoder` dan `to_categorical`.

## 5. Arsitektur Model (LSTM)

Model ini dibangun menggunakan Keras Functional API.

| Layer | Tipe | Output Shape | Keterangan |
| :--- | :--- | :--- | :--- |
| 1 | `Input` | (150,) | Menerima sekuens dengan panjang 150. |
| 2 | `Embedding` | (150, 100) | Mengubah indeks kata (vocab: 91.537) menjadi vektor 100 dimensi. |
| 3 | `Dropout` | (150, 100) | Mencegah *overfitting* dengan me-nonaktifkan 50% neuron. |
| 4 | `LSTM` | (150, 150) | Lapisan LSTM dengan 150 unit. `return_sequences=True`. |
| 5 | `Dropout` | (150, 150) | Mencegah *overfitting* (50%). |
| 6 | `GlobalMaxPooling1D`| (150,) | Mengambil nilai maksimum dari output LSTM. |
| 7 | `Dense` | (64,) | *Fully connected layer* dengan 64 neuron dan aktivasi 'relu'. |
| 8 | `Dropout` | (64,) | Mencegah *overfitting* (50%). |
| 9 | `Dense` (Output) | (2,) | Layer output dengan 2 neuron (untuk 2 kelas) dan aktivasi 'softmax'. |

### Kompilasi Model

* **Optimizer:** `Adam` (dengan *learning rate* 0.0001).
* **Loss Function:** `categorical_crossentropy`.
* **Metrics:** `accuracy`.

## 6. Pelatihan & Evaluasi

### 6.1. Pelatihan

* Model dilatih selama **15 epoch** menggunakan data latih dan divalidasi menggunakan data uji.
* Kurva *loss* dan *accuracy* menunjukkan model konvergen dengan baik.

### 6.2. Hasil Evaluasi

Evaluasi akhir pada data uji memberikan hasil:

* **Test Accuracy:** **97.88%**
* **Test Loss:** 0.0618

### 6.3. Tampilan/Visualisasi Kunci

1.  **Grafik Distribusi Kelas:** Menunjukkan jumlah awal berita "Fake" vs "True".
2.  **Kurva Akurasi & Loss:** Menampilkan progres akurasi dan *loss* model pada data latih dan validasi selama 15 epoch.
3.  **Confusion Matrix:** Menunjukkan performa model secara rinci pada data uji (Prediksi vs. Asli).
