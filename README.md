# Laporan Proyek Machine Learning Predictive Analytics - Rizki Hidayat

## Domain Proyek

Penyakit jantung koroner adalah salah satu penyebab utama kematian di seluruh dunia. Memprediksi penyakit jantung adalah salah satu tugas yang paling menantang di bidang analisis data klinis. *Machine learning (ML)* berguna dalam bantuan diagnostik dalam hal pengambilan keputusan dan prediksi berdasarkan data yang dihasilkan oleh sektor perawatan kesehatan secara global. Teknik ML sudah banyak digunakan dalam bidang medis untuk prediksi penyakit. Dalam hal ini, banyak studi penelitian telah ditunjukkan pada prediksi penyakit jantung menggunakan pengklasifikasi ML.

Menurut Organisasi Kesehatan Dunia [1], CVD adalah penyebab kematian terbesar di dunia, yang mengakibatkan kematian sekitar 17,9 juta orang setiap tahunnya. 

Industri perawatan kesehatan menghasilkan banyak data mengenai pasien, penyakit, dan diagnosis, tetapi tidak dianalisis dengan benar, sehingga tidak memiliki dampak yang sama seperti yang seharusnya pada kesehatan pasien[1]

CVD meliputi arteri koroner, penyakit jantung rematik, penyakit pembuluh darah, dan berbagai masalah jantung dan pembuluh darah. Empat dari setiap lima kematian akibat CVD disebabkan oleh stroke atau serangan jantung. Di antara total kematian, sepertiganya terjadi pada orang yang berusia di bawah 70 tahun [2]

Jenis kelamin, merokok, usia, riwayat keluarga, pola makan yang buruk, kolesterol, kurangnya aktivitas fisik, tekanan darah tinggi, kelebihan berat badan, dan penggunaan alkohol adalah pengaruh risiko utama penyakit jantung. Penyakit jantung juga disebabkan oleh faktor risiko keturunan seperti diabetes dan tekanan darah tinggi [3].

Kelelahan, jantung berdebar, berkeringat, nyeri punggung, nyeri dada, nyeri bahu dan lengan, sesak napas, dan kelemahan secara keseluruhan adalah gejala yang paling umum. Tanda yang paling sering muncul dari kurangnya aliran darah ke jantung adalah nyeri dada. Dalam istilah medis, nyeri dada jenis ini dikenal sebagai Angina [4]. Ada beberapa pemeriksaan yang tersedia untuk membantu mendiagnosis penyakit ini, seperti sinar-X, pemindaian MRI, dan angiografi. Namun, ada kalanya terjadi kekurangan sumber daya dalam keadaan darurat karena tidak tersedianya peralatan medis. Pada penyakit kardiovaskular, waktu sama pentingnya dengan setiap momen dalam mendiagnosis dan mengobati penyakit [4].
 
Sehingga perlu dilakukan upaya untuk mengolah data kesehatan terkait penyakit jantung ini, agar dapat dilakukan prediksi dini untuk menentukan apakah pasien memiliki resiko penyakit jantung atau tidak. Perkembangan pesat teknologi AI dan Machine Learning membuat prediksi dini tersebut dapat dilakukan dengan menggunakan Model Machine Learning. Pengembangan Model Machine Learning untuk melakukan prediksi resiko penyakit jantung ini telah banyak dilakukan oleh peneliti
  
## Business Understanding
Berdasarkan informasi dari WHO, terdapat banyak faktor yang dapat mempengaruhi risiko seseorang terkena penyakit jantung atau tidak.

Untuk itu penting untuk diketahui faktor apa saja kah yang sangat memperngaruhi risiko penyakit jantung tersebut.

Sehingga kedepannya, resiko penyakit jantung tersebut dapat dideteksi secara dini, dan dapat menyelamatkan nyawa pasien.

### Problem Statements

Berdasarkan kondisi tersebut maka perlu dikembangkan sistem prediksi untuk dapat menjawab permasalahan tersebut:
- Dari serangkaian fitur yang ada, fitur apa yang peling berpengaruh terhadap risiko penyakit jantung?
- Apakah resiko penyakit jantung pasien dapat dideteksi secara dini?

### Goals

Untuk menjawab pertanyaan tersebut akan dibuatkan Model *Predictive Analysis* untuk malasah Klasifikasi dengan tujuan sebagai berikut :
- Mengetahui fitur yang paling berpengaruh terhadap risiko penyakit jantung
- Membuat model *machine learning* yang dapat mendeteksi risiko penyakit jantung dengan akurasi terbaik


### Solution statements

Untuk dapat menyelesaikan masalah tersebut dilakukan *modeling* dengan menggunakan 5 algoritma dan melakukan *tuning Hyperparameter* degan bantuan fungsi *GridSearchCV*.

5 algoritma yang digunakan antara lain:
1. *Random Forest Calssifier*
2. *K-Nearest Neighbors*
3. *Gaussian Naive Bayes*
4. *Ada Boost*, dan
5. *XG Boost*

## Data Understanding
Dataset yang digunakan untuk Tugas kali ini adalah *UCI Heart Disease Dataset* dari Kaggle.

Dataset tersebut mengacu pada data dari UCI Machine Learning Repository

Dataset tersebut dibuat oleh: 
1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

Para pembuat database meminta agar setiap publikasi yang dihasilkan dari penggunaan data tersebut menyertakan nama-nama peneliti utama yang bertanggung jawab atas pengumpulan data di setiap institusi. Mereka adalah:

1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D.


Link Dataset:
 1. [Kaggle UCI Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).
 2. [Original UCI Heart Disease Data](https://archive.ics.uci.edu/dataset/45/heart+disease)
    

Dataset terdiri dari 920 baris dengan 16 kolom (fitur) dan terdiri 2 jenis data yaitu data numerik dan data kategori. Jumlah fitur numerik adalah 8 fitur dengan 4 fitur memiliki type data int64 dan 4 fitur memiliki type data float64, sedangkan jumlah fitur kategori adalah 8 fitur dengan type data object. Dari 16 fitur tersebut terdapat 13 fitur parameter medis yang akan menjadi inputan model (mengecualikan kolom id, dataset, dan target (*num*). Detail dari variabel yang terdapat dalam dataset adalah sebagai berikut:

### Variabel-variabel *UCI Heart Disesase dataset* adalah sebagai berikut:
1. id : Nomor Identitas unik untuk setiap pasien
2. age : Usia dari pasien dalam tahun
3. sex : Jenis kelamin dari pasien. Terdiri 2 data yaitu *Male* atau jenis kelamin Pria dan *Female* atau jenis kelamin Wanita
4. dataset : Lokasi dari studi yang dilakukan. Terdiri dari 4 lokasi studi yaitu *Cleveland*, *Hungary*, *Switzerland*, dan *VA Long Beach*
5. cp : Jenis dari sakit dada. Terdiri dari 4 data yaitu *typical angina*, *atypical angina*, *non-anginal*, *asymptomatic*
6. trestbps : Tekanan darah saat istirahat. Nilai dalam satuan mm Hg
7. chol : Kolestrol serum. Nilai dalam satuan  mg/dl
8. fbs : Kadar Gula darah puasa > 120 mg/dl. Terdiri dari 2 data yaitu *True* jika pasien memiliki gula darah puasa > 120 mg/dl, dan *Flase* jika pasien tidak memilikinya
9. restecg : Hasil electrocardiographic istirahat. Terdiri dari 3 data yaitu normal, *st-t abnormality*, dan *lv hypertrophy*
10. thalach : Detak jantung maksimal yang diperoleh
11. exang : *Angina* yang disebabkan oleh latihan. Terdiri 2 data yaitu *True* jika pasien memilikinya dan *False* jika pasien tidak memilikinya
12. oldpeak : Depresi ST yang disebabkan oleh olahraga relatif terhadap istirahat
13. slope : Kemiringan dari puncak segmen *exercise ST*. Terdiri dari 3 data yaitu *upsloping*, *flat*, dan *downsloping*
14. ca : Jumlah dari pembuluh darah *major* (0-3) diwarnai oleh *fluoroscopy*
15. thal : *Thalassemia*. Terdiri dari 3 data yaitu normal, *fixed defect* dan *reversible defect*
16. num: atribut target prediksi terdiri dari 5 data. Terdiri dari 5 data 0 mengindikasi tidak mengalami penyakit jantung, dan 1,2,3,4 mengindikasikan tingkatan penyakit jantung


Dari info dataset yang diperoleh terdapat beberapa fitur yang tidak memiliki jumlah data 920 baris, hal ini mengindikasikan bahwa terdapat *missing values* pada dataset. Karena kolom id dan dataset bukan merupakan parameter medis, maka kedua kolom tersebut dihapus.


Gambaran dari data set adalah sebagai berikut


Tabel 1. Dataset awal 

|     | age |    sex |              cp | trestbps |  chol |   fbs |          restecg | thalch | exang | oldpeak |       slope |  ca |              thal | num |
|----:|----:|-------:|----------------:|---------:|------:|------:|-----------------:|-------:|------:|--------:|------------:|----:|------------------:|----:|
|   0 |  63 |   Male |  typical angina |    145.0 | 233.0 |  True |   lv hypertrophy |  150.0 | False |     2.3 | downsloping | 0.0 |      fixed defect |   0 |
|   1 |  67 |   Male |    asymptomatic |    160.0 | 286.0 | False |   lv hypertrophy |  108.0 |  True |     1.5 |        flat | 3.0 |            normal |   2 |
|   2 |  67 |   Male |    asymptomatic |    120.0 | 229.0 | False |   lv hypertrophy |  129.0 |  True |     2.6 |        flat | 2.0 | reversable defect |   1 |
|   3 |  37 |   Male |     non-anginal |    130.0 | 250.0 | False |           normal |  187.0 | False |     3.5 | downsloping | 0.0 |            normal |   0 |
|   4 |  41 | Female | atypical angina |    130.0 | 204.0 | False |   lv hypertrophy |  172.0 | False |     1.4 |   upsloping | 0.0 |            normal |   0 |
| ... | ... |    ... |             ... |      ... |   ... |   ... |              ... |    ... |   ... |     ... |         ... | ... |               ... | ... |
| 915 |  54 | Female |    asymptomatic |    127.0 | 333.0 |  True | st-t abnormality |  154.0 | False |     0.0 |         NaN | NaN |               NaN |   1 |
| 916 |  62 |   Male |  typical angina |      NaN | 139.0 | False | st-t abnormality |    NaN |   NaN |     NaN |         NaN | NaN |               NaN |   0 |
| 917 |  55 |   Male |    asymptomatic |    122.0 | 223.0 |  True | st-t abnormality |  100.0 | False |     0.0 |         NaN | NaN |      fixed defect |   2 |
| 918 |  58 |   Male |    asymptomatic |      NaN | 385.0 |  True |   lv hypertrophy |    NaN |   NaN |     NaN |         NaN | NaN |               NaN |   0 |
| 919 |  62 |   Male | atypical angina |    120.0 | 254.0 | False |   lv hypertrophy |   93.0 |  True |     0.0 |         NaN | NaN |               NaN |   1 |

Pada Tabel 1. terlihat bahwa terdapat beberapa kolom dengan data "NaN" yang menandakan bahwa pada kolom tersebut terdapat *missing values*. 

Adapun informasi statistik untuk dataset ini adalah:

Tabel 2. Informasi Statistik dataset

|       |        age |   trestbps |       chol |     thalch |    oldpeak |         ca |        num |
|------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| count | 920.000000 | 861.000000 | 890.000000 | 865.000000 | 858.000000 | 309.000000 | 920.000000 |
|  mean |  53.510870 | 132.132404 | 199.130337 | 137.545665 |   0.878788 |   0.676375 |   0.995652 |
|   std |   9.424685 |  19.066070 | 110.780810 |  25.926276 |   1.091226 |   0.935653 |   1.142693 |
|   min |  28.000000 |   0.000000 |   0.000000 |  60.000000 |  -2.600000 |   0.000000 |   0.000000 |
|   25% |  47.000000 | 120.000000 | 175.000000 | 120.000000 |   0.000000 |   0.000000 |   0.000000 |
|   50% |  54.000000 | 130.000000 | 223.000000 | 140.000000 |   0.500000 |   0.000000 |   1.000000 |
|   75% |  60.000000 | 140.000000 | 268.000000 | 157.000000 |   1.500000 |   1.000000 |   2.000000 |
|   max |  77.000000 | 200.000000 | 603.000000 | 202.000000 |   6.200000 |   3.000000 |   4.000000 |

Dari Tabel 2 terlihat bahwa:
1. *Age*:
   - Usia minimal pada dataset adalah 28
   - Rata - rata usia pasien adalah 54
   - Usia maksimal pada dataset adalah 77

2. *tretbps*:
   Nilai minimal adalah 0, hal tersebut mengindikasikan adanya *missing valeues* atau *outliers* pada fitur ini

3. *chol*:
   Nilai minimal adalah 0, hal tersebut mengindikasikan adanya *missing valeues* atau *outliers* pada fitur ini

4. *oldpeak*:
   Nilai minimal adalah -2.6, hal tersebut mengindikasikan *outliers* pada fitur ini


Untuk lebih memahami dataset yang digunakan, dilakukan Exploratory Data Analysis pada data sebagai berikut:

### Exploratory Data Analysis
Tahap yang pertama adalah melakukan Univariate Analysis untuk fitur kategori dan fitur numerik.

Setelah melakuakn Univariate Analysis, selanjutnya melakukan Bivariate Analysis dengan menbandingkan fitur kategori dan fitur numerik dengan target

#### *Univariate Analysis* - Fitur Kategori
Analisis ini dilakukan dengan menggunakan bar plot untuk fitur kategori.
Fitur - fitur kategori pada dataset ini adalah : [*'sex'*,'cp','fbs',*'restecg'*,*'exang'*,*'slope'*,'ca','thal']

##### *Sex* (Jenis kelamin)
Barplot untuk fitur *'sex'* adalah :

![bar-plot-sex](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/c6810fd7-1669-4581-b729-316b7b244292)

Gambar 1. Barplot fitur *sex*


Dari Gambar 1 dapat diketahui bahwa dataset didominasi oleh pasien Pria, dengan jumlah 551 pasein dan perentase 75,7%.

##### Cp (Jenis sakit dada)
Barplot untuk fitur 'cp' adalah :

![bar-plot-cp](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/a96d2922-b657-4456-8c1c-335239bc6a95)

Gambar 2. Barplot fitur cp

Pada Gambar 2, 48,8% pasien memiliki sakit dada *asymptomatic* 22,9% pasien memiliki sakit dada *non-anginal*, 22,7%  pasien memiliki sakit dada *atypical anginan*, dan sisanya meiliki sakit dada *typical angina*

##### fbs (Kadar gula darah puasa)
Barplot untuk fitur 'fbs' adalah:

![bar-plot-fbs](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/87e71170-88bb-4e7f-b377-8768c6ced4db)

Gambar 3. Barplot fitur fbs

Dari Gambar 3 dapat diketahui bahwa kebanyakan pasien tidak memiliki gula darah puasa > 120 mg/dl, dengan 83,7% pesentase.

##### *restecg* (Hasil elevtrokardiografi saat istirahat)
Barplot untuk fitur 'restecg' adalah:

![bar-plot-restecg](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/8e37eb19-985e-47b9-8563-a9fa249db28f)

Gambar 4. Barplot fitur *restecg*

Dari Gamber 4 dapat diketahui bahwa setengah dari total pasien memiliki hasil normal, dimana 23,6% pasien lainnya memilki hasil *Iv hypertrophy* dan sisanya memiliki hasil *st-t abnormality*.

##### *exang* (*Angina* akibat olah raga)
Barplot untuk fitur '*exang*' adalah:

![bar-plot-exang](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/45520db1-b295-416a-9212-042fe87ed879)

Gambar 5. Barplot fitur *Exang*

Pada Gambar 5 terlihat bahwa 62,1% pasien tidak memiliki *angina* yang diakibatkan dari olehraga, sedangkan sisanya memilikinya.

##### *slope* (kemiringan puncak latihan segmen ST)
Barplot untuk fitur 'slope' adalah:

![bar-plot-slope](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/67a6b810-fe1f-4016-aa01-48c0e966f072)

Gambar 6. Barplot fitur *slope*

Pada Gambar 6 terlihat bahwa mayoritas pasien memiliki kemiringan puncak latihan segmen ST *flat* and *upsloping* dengan 47,9% dan 45,5% persentase, sedangkan sisanya memiliki kemiringan puncak latihan segmen ST *downsloping*.

##### ca (jumlah pembuluh darah utama diwarnai oleh flouroskopi)
Barplot untuk fitur 'ca' adalah:

![bar-plot-ca](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/b4a5e5ba-d631-4fcf-86fe-bb359631564b)

Gambar 7, Barplot fitur ca

Dari Gambar 7 dapat diketahui bahwa kebanyakan dari pasien memiliki jumlah pembuluh darah utama 0 dengen 70,7% persentase, dimana sisanya 19% memiliki jumlah 1, 7,7% memiliki jumlah  2 dan 2,6% memiliki jumlah 3 pembuluh darah utama.

##### thal (*Thalassemia*)
Barplot untuk fitur 'tal' adalah:

![bar-plot-thal](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/b7bf20ce-50da-4fe3-bb03-e90241604db1)

Gambar 8. Barplot fitur tal

Berdasarkan informasi pada Gambar 8. Kebanyakan pasien memiliki *thalasemia* normal dan *reversible defect* dengan 48,5% and 45,9% persentase, sisanya memilki *thalasemia fixed defect*

#### *Univeariate Analysis* - Fitur Numerik
Analisis ini dilakukan dengan menggunakan histogram untuk fitur numerik.
Fitur - fitur numerik pada dataset ini adalah : ['age','trestbps','chol','thalch','oldpeak']

Data histogram untuk fitur numerik adalah:

![histogram](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/f4a20956-1d75-4593-bb55-e93179eb7e34)

Gambar 9. Histogram fitur numerik

Dari Gambar 9 diatas dapat disimpulkan bahwa:

1. ***Age*  :** Puncak dari data berada di akhir 50 an sampai 60 an
2. ***Trest bps* (Tekanan Darah) :** Data terkonsentrasi pada kisaran 120-140 mmHg
3. **Chol (Serum Kolestrol)  :** Kebanyakan pasien memiliki nilai Kolestrol antara 200 - 300
4. ***Thalch* (Detak jantung maksimal yang dicapai)  :** Mayoritas pasien memperoleh nilai detak jantung 125 - 175 bpm selama tes.
5. ***Oldpeak* (ST depresi yang diakibatkan oleh latihan)  :** Kebanyakan nilai terkonsentrasi pada nilai 0, hal ini mengindikasikan bahwa pasien tidak mengalami ST depresi yang signifikan selama latihan

### *Bivariate Analysis*

Setelah melakukan *univariate analyisis*, selanjutnya dilakukan *bivariate analayisis*.
1. Untuk data numerik: Mengguankan bar plots untuk menunjukkan nilai rata rata dari tiap fitur terhadap target, dan  KDE plots untuk memahami distribusi masing masing fitur terhadap target. Hal ini membantu dalam memahami bagaimana setiap fitur bervariasi antara dua hasil target.

2. Untuk data kategori : Untuk melihat korelasi antara nilai kategori terhadap nilai target, digunakan *Chi-square test of independence*. Uji statistik ini menilai apakah terdapat hubungan yang signifikan antara dua variabel kategori.

#### *Bivariate Analysis* - Data numerik vs Target

Plot untuk *Bivariate Analysis* - Data numerik vs Target adalah sebagai berikut:

![continous-features-vs-target](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/414e7861-b7d2-4448-8e3f-52edf75b4e30)

Ganmbar 10. Fitur numerik vs target

Dari Gambar 10 dapat ditarik kesimpulan:

1. Usia (*age*): Distribusinya menunjukkan sedikit perubahan dengan rata-rata pasien yang menderita penyakit jantung sedikit lebih muda dibandingkan mereka yang tidak menderita penyakit jantung. Usia rata-rata pasien tanpa penyakit jantung lebih tinggi.
2. Tekanan Darah Saat Istirahat (*trestbps*): Kedua kategori menampilkan distribusi yang tumpang tindih dalam plot KDE, dengan nilai rata-rata yang hampir sama, menunjukkan terbatasnya daya pembeda untuk fitur ini.
3. Kolesterol Serum (chol): Distribusi kadar kolesterol untuk kedua kategori tersebut cukup dekat, namun rata-rata kadar kolesterol pada pasien penyakit jantung sedikit lebih rendah.
4. Pencapaian Denyut Jantung Maksimum (*thalach*): Ada perbedaan nyata dalam distribusi. Pasien dengan penyakit jantung cenderung mencapai detak jantung maksimum yang lebih tinggi selama tes stres dibandingkan dengan mereka yang tidak menderita penyakit jantung.
5. ST Depresi (*oldpeak*): Depresi ST yang disebabkan oleh olahraga relatif lebih rendah pada pasien dengan penyakit jantung. Sebarannya mencapai puncaknya mendekati nol, sedangkan kategori non-penyakit memiliki penyebaran yang lebih luas.

Berdasarkan perbedaan visual dalam distribusi dan nilai rata-rata, Denyut Jantung Maksimum (*thalach*) tampaknya memiliki dampak paling besar terhadap status penyakit jantung, diikuti oleh ST Depresi (*oldpeak*) dan Usia (*age*).

Selain menggunakan Bar Plot dan KDE Plot, digunakan juga *Correlation Matrix* untuk melihat korelasi antar data numerik.

![correlation-matrix](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/cb51c2c0-f8f8-4dc6-9c77-39eb3fee7929)

Gambar 11. *Correlation Matrix* data numerik dengan target

Dari Gambar 11 dapat ditarik kesimpulan, bahwa ca ( Jumlah pembuluh darah mayor (0-3) diwarnai oleh *fluoroscopy*) adalah fitur pertama yang memiliki korelasi yang kuat dengan nilai target, dengan *oldpeak* (ST Depression) di urutan kedua dan *age* di urutan ketiga.

#### *Bivariate Analysis* - Data kategori vs Target
​Hasil *Chi-square test of independence* untuk data kategori adalah sebagai berikut:

Chi-square statistic for sex: 65.9822706513826

p-value for sex: 1.5977088721905247e-13

![observed-counts-sex](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/b44a07b8-cff5-444e-a42c-4e97f5db6aa7)

Gambar 12. *Observed count* untuk fitur *sex*

Dari Gambar 12, dapat ditarik kesimpulan bahwa terdapat 170 pasien Pria dengan penyakit jantung Level 1, dan 20 pasien pria dengan penyakit jantung Level 4, sedangkan kebanyakan pasien Pria tidak memiliki penyakit jantung.


Chi-square statistic for cp: 212.4106179794831

p-value for cp: 8.869629556756493e-39

![observed-counts-cp](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/cf826fd3-f048-4490-bd58-c46827c23ee5)

Gambar 13. *Observed count* untuk fitur *cp*

Dari Gambar 13 dapat ditarik kesimpulan bahwa kebanyakan pasien Pria dengan sakit dada *asymptomatic* memiliki penyakit jantung, dengan 141 pasien dengan penyakit jantung Level 1, 53 pasien dengan penyakit jantung Level 2, 51 pasien dengan penyakit jantung Level 3, dan 17 pasien dengan penyakit jantung Level 4. Sementara kebanyakan pasien dengan sakit dada *atypical angina* tidak memiliki penyakit jantung.


Chi-square statistic for fbs: 28.39522630288068

p-value for fbs: 1.03712317210912e-05

![observed-counts-fbs](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/5dc6bbc1-b5f4-4543-b853-0d7fe6d461fc)

Gambar 14. *Observed count* untuk fitur *fbs*

Dari Gambar 14 diketahui bahwa mayoritas pasien tanpa gula darah puasa > 120 mg/dl, tidak memiliki penyakit jantung, tetapi kebanyakan pasien dengan gula darah puasa > 120 mg/dl, memiliki penyakit jantung.


Chi-square statistic for restecg: 41.2837747903622

p-value for restecg: 1.8447365567425365e-06

![observed-counts-restecg](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/8f06ffbc-f7dd-44b2-bef5-84539934b168)

Gambar 15. *Observed count* untuk fitur *restecg*

Dari Gambar 15 diketahui bahwa kebanyakan pasien dengan *restecg* (Hasil Resting Electrocardiographic) normal tidak memiliki penyakit jantung sementara mayoritas pasien dengan *IV Hyperthropy* dan *st-abnormality* memiliki penyakit jantung.

Chi-square statistic for exang: 194.00745483156376

p-value for exang: 7.295567046348765e-41

![observed-counts-exang](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/908a2496-adbb-4d39-aebc-6d425cc33aa1)

Gambar 16. *Observed count* untuk fitur *exang*

Dari Gambar 16 dapat disimpulkan bahwa mayoritas pasien tanpa *exang* (Angina yang timbul karena latihan) tidak memiliki penyakit jantung sementara pasien dengan *exang* memiliki penyakit jantung level 1 (116 pasien)

Chi-square statistic for slope: 281.339671416197

p-value for slope: 3.833652276746616e-56

![observed-counts-slope](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/4ec1b9a5-728f-466b-8878-d22a9a42a158)

Gambar 17. *Observed count* untuk fitur *slope*

Dari Gambar 17 dapat disimpulkan bahwa mayoritas pasien dengan *slope* (kemiringan puncak latihan segmen ST) *flat* memiliki penyakit jantung dari level 1 (142 pasien), level 2 & level 3 ( keduanya 44 pasien) dan level 4 (11 pasien), sementara sebagian pasien dengan *upslopping* tidak memiliki penyakit jantung.

Chi-square statistic for ca: 299.2114011632948

p-value for ca: 6.875267359872916e-57

![observed-counts-ca](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/42178c5b-4100-4d14-a75e-4a56cab59f21)

Gambar 18. *Observed count* untuk fitur *ca*

Dari Gambar 18, dapat disimpulkan bahwa mayoritas pasien dengan pembuluh darah utama 0 tidak memiliki penyakit jantung, sementara pasien dengan pembuluh darah utama 1,2 dan 3 memiliki penyakit jantung


Chi-square statistic for thal: 284.85474944248324

p-value for thal: 6.860971388362645e-57

![observed-counts-thal](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/bce93097-7dee-4ed2-a5bf-37d35e0be8f9)

Gambar 19. *Observed count* untuk fitur *thal*

Dari Gambar 19 dapat disimpulkan bahwa kebanyakan pasein dengan *reversible defect* memiliki penyakit jantung, sementara pasien dengan *normal thal* tidak memiliki penyakit jantung


## Data Preparation
Setelah dilakukan pengecekan pada data set, masih terdapat *missing values* dan *outliers* pada data set.
Maka proses data preparation yang dilakukan adalah sebagai barikut:

1. Menangani *Missing values*, tahapan proses yang dilakukan antara lain:
   - Melakukan inspeksi *missing values* pada dataset
   - Penangan *missing values* dengan metode *Iterative Imputer*
   - Cek hasil penangan *missing values*
2. Menangani *Outliers*, tahapan proses yang dilakukan antara lain:
   - Melakukan inspeksi *outliers* dengan metode IQR
   - Melakukan visualisasi *outliers* dengan *Boxplot*
   - Melakukan penanganan *outliers* satu per satu pada fitur
   - Cek hasil penanganan *outliers*


Adapun detail dari proses-proses tersebut adalah sebagai berikut:

1. Menangani *Missing Values*
   - Inspeksi *missing values* pada dataset
     Sebelum menangani *missing values*, *missing values* pada data perlu di inspeksi terlebih dahulu.
     *Missing values* pada dataset dapat dilihat dengan fungsi *isnull()*. Jumlah *missing values* pada dataset terdapat pada tabel di bawah ini.
     
     Tabel 3. Jumlah *missing values* pada dataset
     
     | Fitur  | Missing Values|
     |:------ |:------:|
     |age     |       0|
     |sex     |       0|
     |cp      |       0|
     |trestbps|      59|
     |chol    |      30|
     |fbs     |      90|
     |restecg |       2|
     |thalch  |      55|
     |exang   |      55|
     |oldpeak |      62|
     |slope   |     309|
     |ca      |     611|
     |thal    |     486|
     |num     |       0|

     dtype: int64

     Dari Tabel 3 diatas, dapat dilihat bahwa terdapat banyak missing values pada dataset yang perlu ditangani. Tiga fitur dengan jumlah *missing values* terbanyak adalah fitur *ca* dengan 611 *missing values*, fitur *thal* dengan 486 *missing values* dan fitur *slope* dengan 309 *missing values*.
     
     Adapun proporsi jumlah *missing values* untuk tiap tiap fitur adalah sebagai berikut:
     
     Tabel 4. Proporsi *missing values* pada dataset
     
     | Fitur  | Persentase|
     |:------ | ------:|
     |ca       |  66.4  |
     |thal     |   52.8 |
     |slope    |   33.6 |
     |fbs      |    9.8 |
     |oldpeak  |    6.7 |
     |trestbps |    6.4 |
     |thalch   |    6.0 |
     |exang    |    6.0 |
     |chol     |    3.3 |
     |restecg  |    0.2 |
     
     dtype: float64
     
     Dari Tabel 4 diatas dapat diketahui bahwa:

     - Terdapat 10 fitur dengan *missing values*.
     - 7 fitur memiliki persentase *missing values* dibawah 10%.
     - 3 fitur memiliki persentase *missing values* yang tinggi (30%, 50%, 60%).

     Dikarenakan jumlah persentase *missing values* yang cukup tinggi, metode yang dipilih untuk menangani *missing values* tersebut adalah metode *terative Imputer* dengan menggunakan model *machine learning* *Random Forest Classifier* dan *Random Forest Regressor*.
   - Penanganan *missing values* dengan metode *Iterative Imputer*
   
     Adapun metode *Iterative Imputer* mengacu pada proses di mana setiap fitur dimodelkan sebagai fungsi dari fitur lainnya, misalnya. masalah regresi di mana nilai yang hilang diperkirakan. Setiap fitur diperhitungkan secara berurutan, satu demi satu, sehingga nilai yang diperhitungkan sebelumnya dapat digunakan sebagai bagian dari model dalam memprediksi fitur berikutnya.

     Hal ini bersifat iteratif karena proses ini diulang beberapa kali, sehingga estimasi nilai yang hilang dapat dihitung dengan lebih baik seiring dengan estimasi nilai yang hilang di seluruh fitur.

     Algoritma regresi yang berbeda dapat digunakan untuk memperkirakan nilai yang hilang untuk setiap fitur, meskipun metode linier sering kali digunakan untuk kesederhanaan. Jumlah iterasi suatu prosedur seringkali dibuat kecil, misalnya 10. Terakhir, urutan fitur yang diproses secara berurutan dapat dipertimbangkan, seperti dari fitur dengan nilai yang hilang paling sedikit ke fitur dengan nilai yang hilang paling banyak.
​
   - Cek hasil penanganan *missing values*
   
     Setelah dilakukan proses *Iterative Imputer* tersebut, hasil pengecekan *missing values* mendapatkan hasil sebagai berikut:
  
     Tabel 5. Jumlah *missing values* setelah dilakukan proses *Iterative imputer*

     | Fitur  | Missing Values|
     |:------ |:------:|
     |age     |       0|
     |sex     |       0|
     |cp      |       0|
     |trestbps|       0|
     |chol    |       0|
     |fbs     |       0|
     |restecg |       0|
     |thalch  |       0|
     |exang   |       0|
     |oldpeak |       0|
     |slope   |       0|
     |ca      |       0|
     |thal    |       0|
     |num     |       0|

     dtype: int64

     *Missing values* adalah masalah umum dalam proyek *machine learning* dan ilmu data. Kegagalan dalam menangani *missing values* dengan benar dapat mengganggu hasil model *machine learning* atau mengurangi akurasi model. Untuk mengatasi hambatan ini, perlu dilakukan penanganan nilai-nilai yang hilang secara hati-hati. Tujuan dari penanganan *missing values* adalah membuat kumpulan data lengkap yang akan menempatkan analisis pada landasan yang kokoh. Mengabaikan data yang hilang dapat secara langsung memengaruhi performa dan keandalan dari model .
​
2. Menangani *Outliers*,
   - Melakukan inspeksi outliers dengan metode IQR
     IQR adalah *interquartile rang*e atau rentang akar kuartil dari sekumpulan data. IQR digunakan dalam analisis statistik untuk membantu menarik kesimpulan mengenai sekumpulan data. IQR lebih sering digunakan daripada range karena IQR tidak menyertakan data paling luar.
     Secara matematis, IQR dapat dirumuskan sebagai berikut:

     $$IQR = Q3 - Q1$$


     Dengan Q1 adalah nilai di antara median dengan data terkecil atau dapat dikatakan *25th Percentile* sedang Q3 adalah nilai di antara median dengan data terbesar atau dapat dikatakan *75th Percentile*. Untuk menentukan *outliers*, perlu di tetapkan nilai batas atas dan batas bawah. Adapun rumus untuk batas atas dan batas bawah adalah:

    
     $$Batas Atas = Q3 + 1.5 * IQR$$
     
     $$Batas Bawah = Q1 - 1.5 * IQR$$
   

     Maka dengan menggunakan perhitungan IQR tersebut, didapat *outliers* pada dataset ini adalah sebagai berikut:
     
     Tabel 6. Jumlah *Outliiers* pada dataset
     
     | Fitur  | Outliers|
     |:------ |:------:|
     |age      |    0  |
     |trestbps |    28 |
     |chol     |   185 | 
     |thalch   |    2 | 
     |oldpeak  |     3 |
     |ca       |    22 | 
     |num      |     0 |
      
     dtype: int64

     Dari Tabel 6 dapat diketahui bahwa terdapat *outliers* pada 5 fitur antara lain:
     1. Fitur *chol* memiliki 185 *outliers*
     2. Fitur *trestbps* memiliki 28 *outliers*
     3. Fitur *ca* memiliki 22 *outliers*
     4. Fitur *oldpeak* memiliki 3 *outliers*
     5. Fitur *thalch* memiliki 2 *outliers
      
   - Melakukan visualisasi *outliers* dengan Boxplot
     Untuk melihat lebih detail sebaran *outliers* pada dataset, dapat dilakukan visualisasi dengan menggunakan *Boxplot*.
     *Boxplot* juga dikenal sebagai *box-and-whisker plot*, adalah grafik yang menunjukkan sebaran data untuk variabel kontinu. Ini adalah metode non-parametrik yang menampilkan variasi sampel populasi statistik tanpa membuat asumsi apa pun tentang distribusi statistik yang mendasarinya.
     Berikut adalah bagian dasar dari *Boxplot*:
     1. Garis tengah dalam kotak menunjukkan median data. Setengah dari data berada di atas nilai ini, dan setengahnya lagi di bawah. Jika datanya simetris maka mediannya berada di tengah kotak. Jika datanya miring, mediannya akan lebih dekat ke atas atau ke bawah kotak.
     2. Bagian bawah dan atas kotak menunjukkan kuantil atau persentil ke-25 dan ke-75. Kedua kuantil ini juga disebut kuartil karena masing-masing kuantil memotong seperempat (25%) data. Panjang kotak merupakan selisih antara kedua persentil tersebut dan disebut *Interquartile Range* (IQR).
     3. Garis yang memanjang dari kotak disebut *whiskers*. *Whiskers* mewakili variasi data yang diharapkan. *Whiskers* memanjang 1,5 kali IQR dari atas dan bawah kotak. Jika data tidak sampai ke ujung *whiskers*, maka *whiskers* meluas ke nilai data minimum dan maksimum.
     4. Jika ada nilai yang berada di atas atau di bawah ujung *whiskers*, nilai tersebut diplot sebagai titik. Titik-titik ini sering disebut *outlier*. Pencilan lebih ekstrem dibandingkan variasi yang diharapkan. Poin data ini layak untuk ditinjau untuk menentukan apakah data tersebut merupakan *outlier* atau kesalahan; *whiskers* tidak akan menyertakan *outlier* ini.
        
     *Boxplot* untuk data yang mengandung outliers adalah sebagai berikut:
     
     ![boxplot-trestbps](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/f3d5c9b0-6e76-48a9-8d5f-d56f0f6a89ce)
  
     Gambar 20. *Boxplot* untuk fitur *trestbps*

     ![boxplot-chol](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/c169f712-d4ac-425d-9e85-42a6c30b7671)

     Gambar 21. *Boxplot* untuk fitur *chol*
  
     ![boxplot-thalch](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/ab9a382f-6be5-44b6-8b7d-dfa0dd43cc3f)

     Gambar 22. *Boxplot* untuk fitur *thalch*
     
     ![boxplot-oldpeak](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/02bbc002-3946-4b2d-96c5-45699b494603)
  
     Gambar 23. *Boxplot* untuk fitur *oldpeak*

     ![boxplot-ca](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/3129a007-a703-475b-8ebe-6d6004b466ac)

     Gambar 24. *Boxplot* untuk fitur *ca*


   - Melakukan penanganan *outliers* satu per satu pada fitur
     Cara yang paling umum adalah dengan menghilangkan atau menghapus nilai nilai *outliers* yang berada di luar batas nilai IQR.
     Hanya saja, dikarenakan data set yang digunakan hanya terdiri dari 900 data, dan jumlah *outliersnya* cukup banyak, maka *Outliers* pada data set ini diolah secara manual.
     1. Penanganan *Outliers* pada fitur *Tresbps*
        
        Sebelum melakukan penanganan, info statistik dari fitur di inspeksi dengan fitur *describe()*. Info statistik dari fitur *trestbps* adalah:
        
        Tabel 7. Info statistik fitur *trestbps* sebelum *outliers* ditangani
     
        | Statistik | Nilai|
        |:------ | ------:|
        |count   | 920.000000|
        |mean    | 132.494054|
        |std     |  18.534371|
        |min     |   0.000000|
        |25%     | 120.000000|
        |50%     | 130.000000|
        |75%     | 140.160000|
        |max     | 200.000000|
        
        Name: trestbps, dtype: float64

        Dari Tabel 7 dapat dilihat bahwa terdapat data dengan nilai 0 di kolom *trestbps*, dikarenakan tekanan darah tidak mungkin memiliki nilai 0. Maka data dengan nilai 0 pada kolom ini bisa di *drop*. Dikarenakan hasil visual Boxplot pada Gambar 20 terdapat nilai 75 diluar batas bawah, maka nilai dibawah 80 bisa di *drop*.

        Info statistik fitur *trestbps* setelah *outliers* ditangani adalah sebagai berikut:
        
        Tabel 8. Info statistik fitur *trestbps* setelah *outliers* ditangani
        
        | Statistik | Nilai|
        |:------ | ------:|
        |count   | 919.000000|
        |mean    | 132.638226|
        |std     |  18.020920|
        |min     |  80.000000|
        |25%     | 120.000000|
        |50%     | 130.000000|
        |75%     | 140.240000|
        |max     | 200.000000|
        
        Name: trestbps, dtype: float64
        
        Jika dilihat pada Tabel 8, nilai minimal data sudah berubah dari 0 menjadi 80. Jumlah dataset setelah penanganan berkurang dari 920 menjadi 919 data.
        
     2. Penanganan *Outliers Thalch*
    
        Berdasarkan informasi dari Gambar 22,nilai pada kolom *thalch* dapat dimulai dari 71, sehingga nilai dibawah 71 dapat dihilangkan

        Hasil dari penanganan *outliers* untuk fitur *thalch* adalah sebagai berikut:
        
        Tabel 9. Info statistik fitur *thalc* setelah *outliers* ditangani
        
        |Statistik | Nilai|
        |:------ | ------:|
        |count   | 914.000000|
        |mean    | 137.706718|
        |std     |  24.877042|
        |min     |  71.000000|
        |25%     | 120.000000|
        |50%     | 140.000000|
        |75%     | 156.000000|
        |max     | 202.000000|

        Name: thalch, dtype: float64
        
        Dari Tabel 9 diatas, data minimal untuk fitur *thalc* adalah 71. Jumlah dataset berkurang dari 919 menjadi 914 data.
        
     5. Penanganan *Outliers Oldpeak*
        Dari Gambar 23, terdapat 3 outliers pada kolom *oldpeak*. *Outliers* ditangani menggunakan metode IQR, dengan *drop* data yang termasuk kedalam *Outliers*
     
     6. Penanganan *Outliers Chol*
        Sebelum melakukan penanganan, info statistik dari fitur di inspeksi dengan fitur *describe()*. Info statistik untuk fitur *chol* adalah sebagai berikut:
        
        Tabel 10. Info statistik fitur *chol* sebelum *outliers* ditangani
        
        | Statistik | Nilai|
        |:------ | ------:|
        |count   | 911.000000|
        |mean    | 200.296191|
        |std     | 108.268402|
        |min     |   0.000000|
        |25%     | 177.500000|
        |50%     | 223.000000|
        |75%     | 267.000000|
        |max     | 603.000000|
        
        Name: chol, dtype: float64

        Dari Tabel 10, terdapat beberapa nili 0 pada kolom *cholestrol*, sehingga nilai cholestrol tidak mungkin bernilai 0.Jumlah nilai 0 pada kolom *cholestrol* dihitung jumlahnya. Hasilnya terdapat 167 nilai 0 pada kolom chol.

        Dari informasi pada Gambar 21, Diputuskan untuk menyaring nilai *chol* antara 126 dan 400. Dan meghapus data dibawah 126 dan diatas 400.
        
        Info statistik fitur *chol* setelah *outliers* ditangani adalah sebagai berikut:
        
        Tabel 11. Info statistik fitur *chol* setelah *outliers* ditangani
        
        |Statistik | Nilai|
        |:------ | ------:|
        |count   | 728.000000|
        |mean    | 242.279986|
        |std     |  48.270612|
        |min     | 126.000000|
        |25%     | 209.000000|
        |50%     | 236.500000|
        |75%     | 274.000000|
        |max     | 394.000000|
        
        Name: chol, dtype: float64
    
        Dari Tabel 11, dapat diketahui bahwa nilai minimal untuk fitur *chol* adalah 126 dengan nilai maksimal 394. Jumlah dataset setelah dilakukan penanganan berkurang menjadi 728 data.
        *Summary* info statistik untuk semua fitur setelah dilakukan penanganan *outliers* adalah sebagai berikut:
        
        Tabel 12. *Summary* info statistik untuk semua fitur 

        |       |        age |   trestbps |       chol |     thalch |    oldpeak |        ca |        num |
|------:|-----------:|-----------:|-----------:|-----------:|-----------:|----------:|-----------:|
| count | 728.000000 | 728.000000 | 728.000000 | 728.000000 | 728.000000 | 728.00000 | 728.000000 |
| mean  |  52.894231 | 132.993365 | 242.408475 | 141.038214 |   0.962451 |   0.43956 |   0.828297 |
| std   |   9.511945 |  17.340035 |  48.105657 |  24.328196 |   1.086853 |   0.77027 |   1.101167 |
| min   |  28.000000 |  94.000000 | 126.000000 |  71.000000 |   0.000000 |   0.00000 |   0.000000 |
| 25%   |  46.000000 | 120.000000 | 209.000000 | 124.135000 |   0.000000 |   0.00000 |   0.000000 |
| 50%   |  54.000000 | 130.000000 | 236.500000 | 142.000000 |   0.600000 |   0.00000 |   0.000000 |
| 75%   |  59.000000 | 140.000000 | 274.000000 | 160.000000 |   1.800000 |   1.00000 |   1.000000 |
| max   |  77.000000 | 200.000000 | 394.000000 | 202.000000 |   4.400000 |   3.00000 |   4.000000 |

    
## Modeling
Algoritma yang digunakan untuk menyelesaikan masalah *Multiclass Classification* ini adalah sebagai berikut:
1. *Random Forest Classifier*
2. *K-Nearest Neighbors*
3. *Gaussian Naive Bayes*
4. *Ada Boost*, and
5. *XG Boost*

Selain itu dilakukan juga *tuning hyperparameter* menggunakan *GridSearchCV*

Dari kelima model ini akan dipilih model dengan akurasi tertinggi.

Tahapan yang dilakukan untuk training model ini antara lain:
1. *Encoding* Fitur Kategori
   Fitur kategori pada dateset di encoding dengan menggunakan Label Encoder

   Data set sebelum dilakukan *Label Encoder* adalah sebagai berikut:
   
   Tabel 13. Dataset sebelum dilakukan proses *Encoding*
   
   | Index | age |   sex  |        cp       | trestbps |  chol |  fbs  |     restecg    | thalch | exang | oldpeak |    slope    |  ca |        thal       | num |
   |:-----:|:---:|:------:|:---------------:|:--------:|:-----:|:-----:|:--------------:|:------:|:-----:|:-------:|:-----------:|:---:|:-----------------:|:---:|
   |   0   |  63 |  Male  |  typical angina |  145.00  | 233.0 |  True | lv hypertrophy | 150.00 | False |  2.300  | downsloping | 0.0 |    fixed defect   |  0  |
   |   1   |  67 |  Male  |   asymptomatic  |  160.00  | 286.0 | False | lv hypertrophy | 108.00 |  True |  1.500  |     flat    | 3.0 |       normal      |  2  |
   |   2   |  67 |  Male  |   asymptomatic  |  120.00  | 229.0 | False | lv hypertrophy | 129.00 |  True |  2.600  |     flat    | 2.0 | reversable defect |  1  |
   |   3   |  37 |  Male  |   non-anginal   |  130.00  | 250.0 | False |     normal     | 187.00 | False |  3.500  | downsloping | 0.0 |       normal      |  0  |
   |   4   |  41 | Female | atypical angina |  130.00  | 204.0 | False | lv hypertrophy | 172.00 | False |  1.400  |  upsloping  | 0.0 |       normal      |  0  |

   Dari Tabel 13, terlihat bahwa untuk fitur kategori masih memiliki *type* data *object*. Hal ini akan menyulitkan model, dikarenakan model hanya mengenali data numerik. Oleh karena itu dilakukan proses *encoding* untuk merubah *type* data *object* tadi menjadi numerik dengan menggunakan fungsi *Label Encoder*.

   Hasil dari Label Encoder adalah sebagai berikut:
   
   Tabel 14. Dataset seteleh dilakukan proses *encoding*
   
   | Index | age | sex | cp | trestbps |  chol | fbs | restecg | thalch | exang | oldpeak | slope |  ca | thal | num |
   |:-----:|:---:|:---:|:--:|:--------:|:-----:|:---:|:-------:|:------:|:-----:|:-------:|:-----:|:---:|:----:|:---:|
   |   0   |  63 |  1  |  3 |  145.00  | 233.0 |  1  |    0    | 150.00 |   0   |  2.300  |   0   | 0.0 |   0  |  0  |
   |   1   |  67 |  1  |  0 |  160.00  | 286.0 |  0  |    0    | 108.00 |   1   |  1.500  |   1   | 3.0 |   1  |  2  |
   |   2   |  67 |  1  |  0 |  120.00  | 229.0 |  0  |    0    | 129.00 |   1   |  2.600  |   1   | 2.0 |   2  |  1  |
   |   3   |  37 |  1  |  2 |  130.00  | 250.0 |  0  |    1    | 187.00 |   0   |  3.500  |   0   | 0.0 |   1  |  0  |
   |   4   |  41 |  0  |  1 |  130.00  | 204.0 |  0  |    0    | 172.00 |   0   |  1.400  |   2   | 0.0 |   1  |  0  |

   Dari Tabel 14 terihat bahwa data kategori sudah memiliki nilai numerik sehingga data sudah siap untuk dilakukan *modeling*.
   
3. *Split* data menjadi data *train* dan data *test*
   Proses selanjutnya adalah proses *split* data menjadi data *train* dan data *test* dengan perbandingan 80:20 menggunakan fungsi "train_test_split". Agar hasil *modeling* dapat divalidasi oleh data *test*.

4. *Train* data dengan 5 algortitma dengan menerapkan *tuning hyperparameter* menggunakan *GridSearchCV*
   Proses selanjutnya adalah melakukan train data dengan menggunakan 5 algoritma serta melakukan *tuning hyperparameter* sebagai berikut:
   
   - ***Random Forest***       : Untuk *Random Forest*, *hyperparameter* yang diguanakan adalah "*model__n_estimators*" dan "*model__max_depth*".
   - ***K-Nearest Neighbors*** : Untuk KNN, *hyperparameter* yang digunakan adalah "*model__n_neighbors*"
   - ***GaussianNB***          : Untuk *GaussianNB* tidak ada *hyperparameter* yang digunakan
   - ***Ada Boost***           : Untuk *Ada Boost*, *hyperparameter* yang digunakan adalah "*model__n_estimators*" dan "*model__learning_rate*"
   - ***XG Boost***            : Untuk *XG Boost*, *hyperparameter* yang digunakan adalah "*model__n_estimators*" dan "*model__learning_rate*"
  
   Model dilatih menggunakan kelima algoritma diatas dengan bantuan *GridSearchCV*, untuk mendapatkan hasil tuning hyperparameter terbaik dari tiap tiap algoritma. *GridSearchCV* akan melatih model menggunakan algoritma yang ditentukan menggunakan semua *hyperparameter* yang ditentukan. *GridSearchCV* akan menggunakan *setting hyperparameter* terbaik untuk ditampilkan sebagai *output*.
   Matriks evaluasi yang digunakan adalah *Accuracy*, *Precision*, *Recall* ,dan *F1 Score*.

Berikut adalah penjelasan singkat dari algoritma yang digunakan dalam proyek ini:

**1. *Random Forest***

   Algoritma *Random Forest* diperkenalkan oleh Leo Breiman dan Adele Cutler. Algoritma ini didasarkan pada konsep *ensemble learning*, yakni proses menggabungkan beberapa pengklasifikasi untuk memecahkan masalah yang kompleks dan untuk meningkatkan kinerja model.

   Random Forest bekerja dalam dua fase. Fase pertama yaitu menggabungkan sejumlah *N decision tree* untuk membuat *Random Forest*. Kemudian fase kedua adalah membuat prediksi untuk setiap *tree* yang dibuat pada fase pertama.

   ![Illustration-of-random-forest-trees](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/90dcb890-b7cd-4ee9-b545-23e0be77d948)

   Gambar 25. Ilutrasi algoritma *Random Forest*

   Dari Gambar 25, terlihat hubungan *Decision Tree* yang terhubung satu sama lain sehingga membentuk sebuah jaringan pepohonan seperti hutan pada umumnya. Itulah mengapa algoritma ini dinamakan *Random Forest*

   Cara kerja algoritma *Random Forest* dapat dijabarkan dalam langkah-langkah berikut:
   - Algoritma memilih sampel acak dari dataset yang disediakan.
   - Membuat *decision tree* untuk setiap sampel yang dipilih. Kemudian akan didapatkan hasil prediksi dari setiap *decision tree* yang telah dibuat.
   - Dilakukan proses *voting* untuk setiap hasil prediksi. Untuk masalah klasifikasi menggunakan *modus* (nilai yg paling sering muncul), sedangkan untuk masalah regresi akan menggunakan *mean* (nilai rata-rata).
   - Algoritma akan memilih hasil prediksi yang paling banyak dipilih (vote terbanyak) sebagai prediksi akhir.
  
   **Kelebihan Algoritma *Random Forest***
   Berikut adalah kelebihan dari algoritma *Random Forest*:

   - Kuat terhadap data outlier (pencilan data).
   - Bekerja dengan baik dengan data *non-linear*.
   - Risiko *overfitting* lebih rendah.
   - Berjalan secara efisien pada kumpulan data yang besar.
   - Akurasi yang lebih baik daripada algoritma klasifikasi lainnya.
     
   **Kekurangan Algoritma *Random Forest***
   Adapun kelemahan algoritma *Random Forest* adalah sebagai berikut:

   - *Random Forest* cenderung bias saat berhadapan dengan variabel kategorikal.
   - Waktu komputasi pada dataset berskala besar relatif lambat
   - Tidak cocok untuk metode *linier* dengan banyak fitur *sparse*
     
**2. *K-Nearest Neighbors* (KNN)**
  
   KNN adalah singkatan dari *K-Nearest Neighbor*, sebuah algoritma *machine learning* yang bekerja berdasarkan prinsip bahwa objek yang mirip cenderung berada dalam jarak yang dekat satu sama lain. Dengan kata lain, data yang memiliki karakteristik serupa akan cenderung saling bertetangga dalam ruang fitur (*feature space*).

   Prinsip dasar algoritma KNN mengasumsikan bahwa objek yang mirip akan berada dalam jarak yang dekat satu sama lain. Dengan kata lain, data yang memiliki karakteristik serupa akan cenderung terletak berdekatan. KNN menggunakan seluruh data yang tersedia dalam pengambilan keputusan. Ketika ada data baru yang perlu diklasifikasikan, algoritma mengukur tingkat kemiripan atau fungsi jarak antara data baru tersebut dengan data yang sudah ada. Data baru kemudian ditempatkan dalam kelas yang paling banyak dimiliki oleh data tetangga terdekatnya.
      
  
   ![algoritma-knn](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/200b584d-734b-48a0-9151-1e8e50e7f856)

   Gambar 26. Ilustrasi algoritma *K-Nearest Neighbors* (KNN)


   **Kelebihan KNN**
   - Kemudahan implementasi 
   - Kemampuan beradaptasi 
   - *Hyperparameter* yang sedikit

   **Kekurangan KNN**
   - Tidak cocok untuk dataset berukuran besar
   - Tidak cocok untuk dimensi tinggi
   - Penskalaan fitur diperlukan
   - Sensitif terhadap *noise*, *missing value*, dan *outlier*

**3. *Gaussian Naive Bayes***
  
   *Gaussian Naive Bayes* (GNB) adalah teknik klasifikasi yang digunakan dalam *machine learning* berdasarkan pendekatan probabilistik dan distribusi *Gaussian*. *Gaussian Naive Bayes* mengasumsikan bahwa setiap parameter, disebut juga fitur atau prediktor, memiliki kapasitas independen dalam memprediksi variabel keluaran.
   Kombinasi prediksi untuk semua parameter merupakan prediksi akhir yang mengembalikan probabilitas variabel dependen untuk diklasifikasikan dalam setiap kelompok. Klasifikasi akhir diberikan kepada kelompok dengan probabilitas lebih tinggi.

   Distribusi *Gaussian* disebut juga distribusi normal . Distribusi normal adalah model statistik yang menggambarkan sebaran variabel acak kontinu di alam dan ditentukan oleh kurva berbentuk lonceng. Dua ciri terpenting dari distribusi normal adalah *mean* ( $\mu$ ) dan *standard deviasi* ( $\sigma$ ). Rata-rata adalah nilai rata-rata suatu distribusi, dan deviasi standar adalah “lebar” distribusi di sekitar rata-rata.
Variabel ( $X$ ) yang berdistribusi normal terdistribusi secara kontinyu (variabel kontinu) dari $−\infty < X < +\infty$ , dan luas area di bawah kurva model adalah 1.

   ![Illustration-of-how-a-Gaussian-Naive-Bayes-GNB-classifier-works-For-each-data-point](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/cdbf2a36-8753-45ee-819f-0cc8d38e6ea0)

   Gambar 27. Ilustrasi algoritma *Gaussian Naive Bayes*
   
   **Kelebihan *Gaussian Naive Bayes***
   Berikut ini adalah beberapa kelebihan menggunakan pengklasifikasi *Naive Bayes*:
   - Menangani kuantitatif dan data diskrit
   - Kokoh untuk titik *noise* yang diisolasi, misalkan titik yang dirata – ratakan ketika mengestimasi peluang bersyarat data.
   - Hanya memerlukan sejumlah kecil data pelatihan untuk mengestimasi parameter (rata – rata dan variansi dari variabel) yang dibutuhkan untuk klasifikasi.
   - Menangani nilai yang hilang dengan mengabaikan instansi selama perhitungan estimasi peluang
   - Cepat dan efisiensi ruang
   - Kokoh terhadap atribut yang tidak relevan

   **Kelemahan *Gaussian Naive Bayes***
   Berikut ini adalah beberapa kekurangan dari penggunaan pengklasifikasi Naive Bayes:
   - Tidak berlaku jika probabilitas kondisionalnya adalah nol, apabila nol maka probabilitas prediksi akan bernilai nol juga
   - Mengasumsikan variabel bebas

**4. *Ada Boost***
  
   Algoritma *AdaBoost*, singkatan dari *Adaptive Boosting*, adalah sebuah teknik *Boosting* yang digunakan sebagai metode *ensemble** dalam machine learning*. Algoritma *AdaBoost* bersifat iteratif atau berulang. Cara kerja algoritma ini dimulai dengan melatih sebuah *weak classifier* pada data pelatihan.

   *Weak classifier* kemudian diberi bobot berdasarkan performanya. Selanjutnya, algoritma melatih *weak classifier* kedua menggunakan data yang telah diberi bobot. *Weak classifier* kedua kemudian diberi bobot berdasarkan performanya. Proses ini diulang sejumlah iterasi tertentu atau hingga tingkat kesalahan berada di bawah ambang batas yang ditentukan. *Classifier* akhir adalah rata-rata terbobot dari semua *weak classifiers*. Bobot ditentukan berdasarkan tingkat kesalahan dari masing-masing *weak classifier*. Semakin rendah tingkat kesalahan, semakin tinggi bobotnya.

   ![Ada-Boost](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/92e44689-6fb7-4ff0-bde6-57c8e480ed5a)

   Gambar 28. Ilustrasi algoritma *Ada Boost*


   Secara runtun, cara kerja algoritma ini dapat dijabarkan sebagai berikut:
   1. Inisialisasi bobot sampel pelatihan
      Langkah pertama adalah memberikan bobot yang sama pada semua sampel pelatihan. Bobot ini digunakan untuk memberikan penekanan pada contoh-contoh yang salah diklasifikasikan pada iterasi-iterasi berikutnya.
   2. Melatih *weak classifier*
      *Weak classifier* dilatih pada data pelatihan yang telah diberi bobot pada setiap iterasi. Tujuan dari *weak classifie*r adalah untuk mengklasifikasikan sampel sebagai positif atau negatif. Ada beberapa jenis *weak classifier* yang dapat digunakan, seperti *decision tree*, model linear, atau *support vector machine*.
   3. Evaluasi performa *weak classifier*
      Setelah *weak classifier* dilatih, performanya dievaluasi pada data pelatihan. Sampel-sampel yang salah diklasifikasikan diberikan bobot yang lebih tinggi untuk memberi prioritas pada iterasi berikutnya.
   4. Memperbarui bobot sampel pelatihan
      Bobot sampel pelatihan diperbarui berdasarkan klasifikasinya oleh *weak classifier*. Bobot sampel yang salah diklasifikasikan ditingkatkan, sedangkan bobot sampel yang benar diklasifikasikan dikurangi. Hal ini memastikan bahwa *weak classifier* fokus pada sampel yang lebih sulit.
   5. Ulangi langkah 2-4 sesuai jumlah iterasi yang ditentukan
      Langkah-langkah sebelumnya diulang sesuai jumlah iterasi yang ditentukan, atau hingga mencapai ambang batas tertentu. Bobot sampel pelatihan disesuaikan pada setiap iterasi untuk memberi prioritas pada sampel yang salah diklasifikasikan dan belajar dari kesalahan yang dilakukan oleh *weak classifier*.
   6. Menggabungkan *weak classifier* menjadi sebuah model yang kuat
      Setelah jumlah iterasi yang ditentukan, *weak classifier* digabungkan menjadi sebuah model yang kuat menggunakan jumlah tertimbang dari keluaran mereka. Model akhir dapat melakukan prediksi yang akurat pada data baru yang tidak terlihat sebelumnya.

   **Kelebihan Algoritma *AdaBoost***
   Berikut beberapa kelebihan algoritma *AdaBoost*:
   - Meningkatkan performa prediksi
     *Adaboost* dapat secara signifikan meningkatkan akurasi prediksi dalam pemodelan *machine learnin*g. Dengan menggabungkan *weak learners* menjadi *strong learner*, *Adaboost* dapat mengatasi kesalahan klasifikasi dan meningkatkan kemampuan prediksi model.
   - Penanganan data yang kompleks
     *Adaboost* efektif dalam menangani data yang kompleks dan memiliki interaksi fitur yang rumit. Dalam kasus di mana hubungan antara fitur-fitur input dan variabel output tidak sederhana, *Adaboost* dapat menangkap pola yang lebih kompleks daripada *weak learners* individu.
   - Mencegah *overfitting*
     Dengan memberikan bobot pada contoh-contoh yang salah diklasifikasikan, *Adaboost* dapat mengurangi risiko *overfitting*. Hal ini membantu model untuk tidak terlalu fokus pada contoh-contoh pelatihan yang sulit dan memperbaiki generalisasi pada data uji.
     
   **Kelemahan Algoritma *AdaBoost***
   Algoritma *Adaboost* juga memiliki kelemahan seperti:
   - Sensitif terhadap *noise*
     *Adaboost* cenderung sensitif terhadap data yang mengandung *noise* atau *outlier*. Kehadiran contoh-contoh yang tidak representatif atau gangguan dapat mempengaruhi pembelajaran dan menghasilkan model yang kurang akurat.
   - Risiko *overfitting* pada data pelatihan yang kecil
     Jika dataset pelatihan sangat kecil, terdapat risiko *overfitting* pada *Adaboost*. Model yang terlalu kompleks dapat dengan mudah "menghafal" contoh-contoh pelatihan dan gagal dalam generalisasi pada data baru.
   - Waktu pelatihan yang lebih lama
     *Adaboost* melibatkan iterasi berulang untuk melatih *weak learners* dan memperbarui bobot contoh-contoh pelatihan. Oleh karena itu, waktu pelatihan algoritma ini cenderung lebih lama dibandingkan dengan beberapa algoritma machine learning lainnya.

**5. *XG Boost***
  
   *XG Boost* adalah salah satu implementasi dari algoritma *Gradient Boosting*. *Gradient Boosting* adalah sebuah teknik yang menggabungkan beberapa model yang lemah (*weak model*) menjadi sebuah model yang kuat. Model-model lemah ini sering disebut dengan *weak learners*, dan dapat berupa model regresi atau klasifikasi sederhana seperti *Decision Tree*. Pada setiap iterasi, *Gradient Boosting* akan menambahkan *weak learner* baru dan mengoreksi prediksi sebelumnya dengan memperhitungkan kesalahan pada prediksi tersebut.

   Secara matematis, *Gradient Boosting* mengoptimalkan suatu fungsi objektif dengan mengevaluasi *gradient* pada setiap titik. Fungsi objektif yang umum digunakan dalam *Gradient Boosting* adalah fungsi *Mean Squared Error (MSE)* untuk regresi dan fungsi *Log-Loss* untuk klasifikasi. Dalam setiap iterasi, *Gradient Boosting* memperbarui *residual error* dengan mengurangi hasil prediksi dari target, lalu menambahkan *weak learner* baru yang menyelesaikan masalah *residual erro*r yang dihasilkan. Dengan cara ini, *Gradient Boosting* membangun sebuah model yang kuat dari beberapa model yang lemah.

   Algoritma *Gradient Boosting* bekerja dengan menggabungkan beberapa model yang lemah menjadi sebuah model yang lebih kuat. Algoritma ini menggunakan pendekatan iteratif, di mana setiap iterasi bertujuan untuk meningkatkan model sebelumnya dengan menambahkan model baru. Proses ini dilakukan secara berulang-ulang hingga model yang dihasilkan memenuhi kriteria tertentu, seperti nilai loss function yang cukup kecil.

   ![A-simple-example-of-visualizing-gradient-boosting](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/0535309f-f1a2-408f-bf8a-ed021111c8dc)

   Gambar 29. Ilustrasi Algortima *XG Boost*


   Proses iteratif dalam algoritma *Gradient Boosting* terdiri dari beberapa tahap, yaitu:
   1. Inisialisasi model: Tahap pertama dalam algoritma *Gradient Boosting* adalah inisialisasi model. Pada tahap ini, model awal dibuat sebagai model konstan yang merupakan rata-rata atau *median* dari target variable.
   2. Membuat weak model: Pada tahap ini, weak model dibuat sebagai model yang mampu memprediksi error dari model sebelumnya. Model lemah biasanya berupa *decision tree* yang dangkal dengan satu atau dua percabangan.
   3. Menghitung *residual error*: Setelah model lemah dibuat, *residual error* dihitung sebagai selisih antara nilai prediksi dari model sebelumnya dan nilai asli dari target variable.
   4. Menyusun kembali data *training*: Pada tahap ini, data *training* diubah dengan menggunakan *residual error* sebagai target variable.
   5. Membuat model baru: Pada tahap ini, model baru dibuat dengan memprediksi *residual error* yang dihasilkan dari model sebelumnya.
   6. Menggabungkan model: Model baru yang dibuat pada tahap sebelumnya digabungkan dengan model sebelumnya untuk membentuk model yang lebih baik.
   7. Iterasi berulang: Tahap-tahap di atas diulang berulang-ulang hingga mencapai kondisi berhenti yang ditentukan, seperti jumlah iterasi yang telah ditentukan atau ketika model tidak mengalami peningkatan yang signifikan lagi.
     
   Setelah iterasi selesai dilakukan, model yang dihasilkan akan digunakan untuk memprediksi nilai target pada data testing yang baru.

   Berikut adalah kelebihan dan kekurangan dari algoritma *Gradient Boosting*:
   **Kelebihan *Gradient Boosting***
   - Akurasi yang tinggi: *Gradient Boosting* sering menghasilkan model yang akurat dan kuat, terutama ketika digunakan pada data yang kompleks dan tidak terstruktur.
   - Tidak memerlukan persyaratan data yang ketat: Algoritma ini dapat digunakan pada berbagai jenis data tanpa memerlukan asumsi yang ketat, seperti asumsi tentang distribusi data atau homoskedastisitas.
   - Kecepatan komputasi yang cepat: Beberapa implementasi dari *Gradient Boosting*, seperti *XGBoost* dan *LightGBM*, dapat digunakan untuk mempercepat waktu komputasi dengan teknik-teknik seperti parallel computing dan caching.
   **Kekurangan *Gradient Boosting***
   - Memerlukan tuning yang cermat: Algoritma ini memerlukan tuning parameter yang cermat untuk mendapatkan model yang optimal. Hal ini dapat memakan waktu dan mengharuskan penggunaan *cross-validation* dan teknik tuning parameter lainnya.
   - Mudah *overfitting*: *Gradient Boosting* dapat cenderung *overfit* pada data *training* jika tidak dilakukan pengaturan parameter yang baik. *Overfitting* terjadi ketika model terlalu kompleks dan terlalu menyesuaikan dengan data *training*, sehingga tidak dapat melakukan generalisasi dengan baik pada data yang belum pernah dilihat sebelumnya.
   - Memerlukan data yang besar: *Gradient Boosting* memerlukan jumlah data yang besar untuk memperoleh model yang akurat dan stabil. Jika jumlah data terlalu sedikit, algoritma ini dapat menjadi tidak stabil dan menghasilkan model yang tidak akurat.
   
Setelah melakukan training model dengan kelima algoritma tersebut didapat model yang terbaik adalah **Model *Ada Boost***. Model Random Forest dipilih karena memiliki nilai acccuracy tertinggi yaitu 0.74. Adapun rangkuman nilai accuracy dari tiap tiap algoritma adalah sebagai berikut:

Tabel 15. Hasil metric seluruh algoritma

| Index |        Model        | accuracy | precision | recall |  f1  |
|:-----:|:-------------------:|:--------:|:---------:|:------:|:----:|
|   0   |      Ada Boost      |   0.74   |    0.74   |  0.74  | 0.71 |
|   1   |      GaussianNB     |   0.62   |    0.70   |  0.62  | 0.65 |
|   2   | K-Nearest Neighbors |   0.70   |    0.70   |  0.70  | 0.69 |
|   3   |    Random Forest    |   0.73   |    0.75   |  0.73  | 0.73 |
|   4   |       XG Boost      |   0.70   |    0.72   |  0.70  | 0.70 |

Dari Tabel 15 dapat dilihat bahwa *Ada Boost* memiliki nilai accuracy yang tinggi yaitu 0,74. Maka model terbaik untuk data set ini adalah **Model Ada Boost** dengan nilai metriks sebegai berikut:

Tabel 16. Nilai metrik untuk *Ada Boost*

| Model:      | Ada Boost     |
|-------------|:-------------:|
| Accuracy:   |      0.74     |
| Precision:  |      0.74     |
| Recall:     |      0.74     |
| F1:         |      0.71     |



## Evaluation
Metrik yang digunakan untuk permasalahan *Multiclass Classification* ini adalah *Accuracy*, *Precision*, *Recall*, dan *F1 Score*. Penjelasan untuk masing masing metrik adalah sebagai berikut:

1. *Accuracy*
   
   *Accuracy* adalah metrik yang mengukur seberapa sering model *machine learning* memprediksi hasil dengan benar. Anda dapat menghitung akurasi dengan membagi jumlah prediksi yang benar dengan jumlah total prediksi.

   Formula dari *Accuracy* adalah sebagi berikut:

   
   $$Accuracy = \dfrac{Prediksi Benar}{Total Prediksi}$$
   

   *Accuracy* dapat diukur pada skala 0 hingga 1, atau sebagai persentase. Semakin tinggi akurasinya, semakin baik. Anda dapat mencapai akurasi sempurna 1,0 ketika setiap prediksi yang dibuat model benar.

   **Cara kerja *Accuracy***
   Sebagai contoh, terdapat model *machine learning* untuk melakukan deteksi *email spam*.

   Untuk setiap *email* di dalam dataset, sistem ini menghasilkan prediksi dan menetapkan salah satu dari dua kelas: "*spam*" atau "bukan *spa*m". Berikut ini adalah bagaimana kita dapat memvisualisasikan label yang diprediksi:

   ![Accuracy-Labeled-Data](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/e8cac743-2db4-4039-a721-3ededb907bb8)

   Gambar 30. Data yang telah dilabeli


   Setelah mendapatkan label data sebenarnya (mengetahui email mana yang merupakan *spam* dan mana yang bukan), dapat dilakukan evaluasi apakah prediksi model sudah tepat. Visualisasi hasil prediksi yang benar dan salah oleh model :

   ![Accuracy-Hasil-Prediksi](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/5378bf2d-c06a-4118-b571-30c77819b1ba)

   Gambar 31. Hasil Prediksi


   Maka, *accuracy* dapat dihitung dengan cara membagi jumlah prediksi yang benar oleh model dengan total prediksi. Pada contoh kasus ini 52 dari 60 prediksi (ditandai dengan tanda centang hijau) adalah prediksi benar. Artinya model memiliki *accuracy* 87%.

   ![Accuracy-Perhitungan-Akurasi](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/390841f0-6b77-4607-90be-8959e2662d89)

   Gambar 32. Perhitungan *Accuracy*

   
   Berikut adalah kelebihan dan kekurangan dari Metrik *Accuracy*
   
   **Kelebihan**
   - Akurasi adalah metrik yang sangat membantu ketika Anda berurusan dengan kelas-kelas yang seimbang dan peduli dengan "ketepatan" model secara keseluruhan, dan bukan kemampuan untuk memprediksi kelas tertentu.
   - Akurasi mudah dijelaskan dan dikomunikasikan.
  
   **Kekurangan**
   - Jika Anda memiliki kelas yang tidak seimbang, akurasi kurang berguna karena memberikan bobot yang sama pada kemampuan model untuk memprediksi semua kategori.
   - Mengkomunikasikan akurasi dalam kasus seperti itu dapat menyesatkan dan menyamarkan kinerja yang rendah pada kelas target.

2. *Precision*

   *Precision* adalah metrik yang mengukur seberapa sering model *machine learning* memprediksi kelas positif dengan benar. Anda dapat menghitung presisi dengan membagi jumlah prediksi positif yang benar (*true positive*) dengan jumlah total contoh yang diprediksi oleh model sebagai positif (*true dan false positive*).

   Secara matematis metrik Precision dapat dihitung dengan persamaan berikut:
   
   $$Precision = \dfrac{True Positive}{True Positive + False Positive}$$
   

   Precision dapat diukui menggunakan skala 0 hingga 1, atau sebagai persentase. Semakin tinggi presisi, semakin baik. Anda dapat mencapai presisi sempurna 1,0 ketika model selalu tepat ketika memprediksi kelas target: model tidak pernah menandai kesalahan apa pun.

   **Cara Kerja**
   Contoh kasus jika terdapat masalah yang tidak seimbang, seperti *spam* yang hanya terjadi pada 5% dari semua *email*, dari 60 email, hanya 3 email yang benar benar *spam*.

   Berikut ini adalah *visualisasi* label yang sebenarnya (pembagian yang sebenarnya antara email *spam* dan *non-spam*).

   ![Precision-Labeled-Data](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/762f2c74-9ab7-44db-978f-024a93885fe6)

    Gambar 33. Data email spam tidak seimbang 


   Dikarenakan jumlah *email spam* yang sedikit, asumsikan model mendeteksi semua *email* bukan *spam* dan tidak dapat mendeteksi *spam*. Maka jika menggunakan metrik *Accuracy*, nilainya adalah 95% (Model benar menebak 57 dari 60 email, tetapi nilai *Precision* adalah 0.

    ![Precision-Hasil-Prediksi](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/fdae07a1-aaef-4699-863f-ae8a9a12e0c3)

   Gambar 34. Hasil prediksi data email tidak seimbang

   Untuk menghitung *Precision*, perlu dilakukan pembagian jumlah *email spam* yang diprediksi dengan benar dengan jumlah totalnya. Namun, jumlah *email spam* yang diidentifikasi dengan benar adalah 0. Ada 3 *email spam* dalam dataset, dan model melewatkan semuanya. Semua prediksi yang benar adalah tentang *email* bukan *spam*.

   Dengan cara ini, metrik *Precision* mengoreksi kelemahan utama dari *Accuracy*. Hal ini dengan jelas mengkomunikasikan bahwa model tidak dapat menyelesaikan masalah.

   Selanjutnya, diasumsikan model dapat mengidentifikasi beberapa *email* sebagai *spam*, kemudian membandingkan prediksi model terhadap label yang sebenarnya dan mendapatkan hasil sebagai berikut:

   ![Precision-Hasil-Prediksi-2](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/019f3dff-d143-4bed-8b50-188480fb9577)

   Gambar 35. Hasil prediksi kedua untuk data *email* tidak seimbang


   Nilai *Precisionnya* adalah Jumlah *email spam* yang diprediksi dengan benar (3) dibagi total prediksi *positive* (6).

   ![Precision-Perhitungan](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/5667308e-4364-4e7b-aa28-0c03a3782554)

   Gambar 36. Perhitungan *Precision*


   Nilai Precisionnya adalah 50% sesuai dengan Gambar 36. Model ini melabeli 6 *email* sebagai *spam* dan benar separuhnya. 3 dari 6 email yang dilabeli sebagai *spam*, pada kenyataannya, adalah *spam* (*true positive*). Tiga lainnya meleset (*false positive*). Ketepatannya adalah 3/(3+3) = 50%.

   Berikut adalah kelebihan dan kekurangan dari Metrik *Precision*
   **Kelebihan**
   - Metrik ini bekerja dengan baik untuk masalah dengan kelas yang tidak seimbang karena menunjukkan ketepatan model dalam mengidentifikasi kelas target.
   - *Precision* berguna ketika biaya *False Negative* tinggi.
  
   **Kekurangan**
   *Precision* tidak mempertimbangkan *False Negative*. Artinya: tidak memperhitungkan kasus-kasus ketika kita melewatkan target event.

4. *Recall*
   
   *Recall* adalah metrik yang mengukur seberapa sering model *machine learning* mengidentifikasi contoh positif (*true positive*) dengan benar dari semua sampel positif yang sebenarnya dalam kumpulan data. Anda dapat menghitung *recall* dengan membagi jumlah *true positive* dengan jumlah contoh positif. Yang terakhir ini mencakup hasil *true positive* (kasus yang berhasil diidentifikasi) dan hasil *false negative* (kasus yang terlewatkan).

   Secara matematis metrik *recall* dapat dihitung dengan persamaan berikut:

   
   $$Recall = \dfrac{True Positive}{True Positive + False Negative}$$
   

   *Recall* dapat diukur menggunakan skala 0 hingga 1 atau sebagai persentase. Semakin tinggi *recall*, semakin baik. Anda dapat mencapai recall sempurna sebesar 1,0 ketika model dapat menemukan semua contoh kelas target dalam dataset.

   **Cara Kerja**
   Dengan menggunakan prediksi kasus data tidak imbang pada prediksi email spam sesuai Gambar 35 berikut:
   
   Untuk kasus ini telah diketahui model *Accuracy* adalah 95% (model dengan benar melabeli 57 dari 60 email) dan model *Precision* sebesar 50% (model dengan benar melabeli 3 dari 6 *email spam*).

   Untuk menghitung *Recall*, jumlah email spam yang ditemukan dibagi dengan jumlah total *email spam* dalam dataset.
   
   ![Recall-Perhitungan](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/9924c978-a14f-4f07-9107-9ce3272dfef8)

   Gambar 37. Perhitungan *recall*


   Maka nilai *Recall* adalah 100%. Ada 3 *email spam* di dalam dataset, dan model menemukan semuanya! *Recall* menghitungnya sebagai 3/(3+0). Tidak ada *False Negative* karena model tidak melewatkan spam.

   Berikut adalah kelebihan dan kekurangan dari Metrik *Recall*
   **Kelebihan**
   - Bekerja dengan baik untuk masalah dengan kelas yang tidak seimbang karena difokuskan pada kemampuan model untuk menemukan objek dari kelas target.
   - *Recall* berguna ketika biaya *false negative* tinggi. Dalam kasus ini, Anda biasanya ingin menemukan semua objek dari kelas target, meskipun hal ini menghasilkan beberapa *false positive* (memprediksi positif padahal sebenarnya negatif).
  
   **Kekurangan**
   Recall tidak memperhitungkan biaya untuk *False Positive*

6. *F1 score*

   *F1 score* adalah matrik *machine learning* yang mengukur akurasi model. Skor ini menggabungkan skor *precision* dan *recall* dari sebuah model. *F1 score* menggabungkan *precision* dan *recall* menggunakan rata-rata harmoniknya, dan memaksimalkan *F1 score* berarti secara bersamaan memaksimalkan *precision* dan *recall*. Oleh karena itu, *F1 score* telah menjadi pilihan para peneliti untuk mengevaluasi model mereka bersama dengan akurasi.

   Secara matematis *F1 score* dapat dihitung dengan persamaan berikut:
   
   $$F1 score = \dfrac{2}{\dfrac{1}{Precision} + \dfrac{1}{Recall}}$$
   

   Mengapa *F1 score* dihitung menggunakan rata-rata harmonik dan bukannya rata-rata aritmatika atau geometris sederhana? Sederhananya: rata-rata harmonik mendorong nilai yang sama untuk *precision* dan *recall*. Artinya, semakin jauh nilai *precision* dan *recall* menyimpang dari satu sama lain, semakin buruk nilai rata-rata harmoniknya.
   Dalam hal empat elemen dasar dari *confusion matrix*, dengan mengganti ekspresi untuk nilai *precision* dan *recall* pada persamaan di atas, *F1 score* juga dapat dituliskan sebagai berikut:

   
   $$F1 score = \dfrac{True Positive}{ True Positive + \dfrac{1}{2} (False Positive + False Negative)}$$
   

   Adapaun Kelebihan dan kekurangan *F1 Score*
   
   **Kelebihan**
   - Menyeimbangkan *precision* dan *recall*: Ini mempertimbangkan *precision* dan *recall* dan memberikan nilai tunggal yang menyeimbangkan pertukaran antara metrik ini. Hal ini berguna untuk mengevaluasi model dengan pertukaran yang berbeda antara *precision* dan *recall*, tergantung pada masalah dan konteks tertentu.
   - Mudah ditafsirkan: Ini adalah metrik yang sederhana dan intuitif yang berkisar antara 0 hingga 1, dengan nilai yang lebih tinggi menunjukkan kinerja yang lebih baik. Sangat mudah untuk dipahami dan ditafsirkan, bahkan untuk pemangku kepentingan non-teknis.
   - Kuat terhadap ketidakseimbangan kelas: Kuat terhadap ketidakseimbangan kelas, yang merupakan masalah umum dalam tugas klasifikasi biner di mana satu kelas jauh lebih sering daripada yang lain. Ini memberikan evaluasi yang seimbang terhadap kinerja model di kedua kelas.
   - Berlaku untuk dataset kecil dan besar: Ini berlaku untuk dataset kecil dan besar dan dapat memberikan evaluasi cepat terhadap kinerja model tanpa memerlukan metrik yang lebih kompleks.
   - Dapat digunakan untuk pemilihan model: Dapat digunakan sebagai kriteria untuk pemilihan model atau penyetelan *hyperparameter*, sehingga memungkinkan perbandingan yang adil antara model atau pengaturan yang berbeda.

   **Kekurangan**
   - *F1 score* tidak memberikan informasi tentang distribusi kesalahan: Nilai ini memberikan nilai tunggal yang merangkum kinerja model di seluruh *precision* dan *recall*. Namun, nilai ini tidak memberikan informasi apa pun tentang distribusi kesalahan, yang dapat menjadi penting untuk aplikasi tertentu.
   - *F1 score* mengasumsikan bahwa *precision* dan *recall* sama pentingnya: Skor ini memberikan bobot yang sama pada *precision* dan *recall*, dengan asumsi keduanya memiliki kepentingan yang sama. Namun, *precision* dan *recall* mungkin memiliki biaya atau signifikansi yang berbeda di beberapa aplikasi, dan metrik lain mungkin lebih tepat.
   - *F1 score* mungkin tidak optimal untuk klasifikasi multikelas: Skor ini dirancang untuk masalah klasifikasi biner dan mungkin tidak dapat diterapkan secara langsung pada masalah klasifikasi multikelas. Metrik lain, seperti akurasi atau *F1 scoare mikro/makro*, mungkin lebih tepat.
   - *F1 score* mungkin tidak sensitif terhadap pola tertentu dalam data: Ini adalah metrik umum yang tidak mempertimbangkan pola atau karakteristik tertentu dari data. Namun, dalam beberapa kasus, metrik yang lebih khusus mungkin diperlukan untuk menangkap sifat-sifat spesifik dari masalah.
   

Dari hasil didapat model terbaik untuk menyelesaikan permasalahan *Multiclass Classification* untuk data set ini adalah **Model Ada Boost** dengan performa seperti pada Tabel 16.

Adapun perbandingan tiap tiap metrik untuk kelima algoritma terlihat pada bar plot berikut:

![model-metrics](https://github.com/Rizki-Kidut/Predictive-Analytics---UCI-Heart-Disease-Data/assets/116653612/065ec581-736d-4856-8a5c-ac24ec3ecaec)

Gambar 38. Barplot metrik seluruh algoritma

Selain itu, dilakukan prediksi terhadap 5 nilai target dan dibandingakan dengan nilai target sesungguhnya. Hasilnya adalah sebagai berikut:

Tabel 17. Hasil prediksi algoritma 

| Index | y_true | prediksi_Random Forest | prediksi_K-Nearest Neighbors | prediksi_GaussianNB | prediksi_Ada Boost | prediksi_XG Boost |
|:-----:|:------:|:----------------------:|:----------------------------:|:-------------------:|:------------------:|:-----------------:|
|   39  |    0   |            0           |               0              |          1          |          0         |         0         |
|  341  |    0   |            0           |               0              |          0          |          0         |         0         |
|  332  |    0   |            0           |               0              |          0          |          0         |         0         |
|  524  |    1   |            1           |               2              |          3          |          1         |         1         |
|  562  |    1   |            1           |               1              |          1          |          1         |         1         |

Dari Tabel 17, dapat ditarik kesimpulan bahwa dari kelima algoritma yang digunakan Algoritma Random Forest, Ada Boost dan XG Boost dapat memprediksi kelima nilai target dengan benar. Sedangkan algoritma KNN dapat memprediksi 4 dari 5 nilai target dan GaussianNB hanya dapat memprediksi 3 dari 5 nilai target.

Dari Tabel 16 dan Tabel 17 dapat ditarik kesimpulan bahwa model *machine learning* yang dibuat sudah dapat mendeteksi penyakit jantung dari fitur fitur pada dataset, meskipun nilai *accuracy*, *precision*, *recall*, dan *F1 score* masih berada dibawah 80%. Hal ini menunjukkan masih perlu dilakukan perbaikan dari *tuning hyperparameter* maupun dari proses data *preparation*.

Jawaban dari *problem statement* adalah sebagai berikut:
- Dari serangkaian fitur yang ada, fitur apa yang peling berpengaruh terhadap risiko penyakit jantung?

  Dari Gambar 11 *Correltion Matrix* dapat ditarik kesimpulan, bahwa ca ( Jumlah pembuluh darah mayor (0-3) diwarnai oleh *fluoroscopy*) adalah fitur pertama yang memiliki korelasi yang kuat dengan nilai target, dengan *oldpeak* (ST Depression) di urutan kedua dan *age* di urutan ketiga.
  
- Apakah resiko penyakit jantung pasien dapat dideteksi secara dini?

  Risiko penyakit jantung dapat dideteksi dengan model yang dibuat (Algoritma terbaik *Ada Boost*), meskipun performa model menunjukkan masih perlu dilakukan perbaikan dari segi *tuning hyperparameter* dan proses pembersihan data agar dapat dihasilkan performa yang lebih baik lagi dalam mendeteksi risiko penyakit jantung


  Format Referensi: 
  1. [World Health Organization Cardiovascular Diseases (CVDs)](https://www.who.int/health-topics/cardiovascular-diseases/#tab=tab_1) 
  2. [AFRO.WHO Cardiovascular Diseases (CVDs)](https://www.afro.who.int/health-topics/cardiovascular-diseases) 
  3. [Heart.org Why High Blood Pressure is a silent killer]( https://www.heart.org/en/health-topics/high-blood-pressure/why-high-blood-pressure-is-a-silent-killer/know-your-risk-factors-for-high-blood-pressure) 
  4. [Balla C., Pavasini R., Ferrari R. Treatment of Angina: Where Are We? *Cardiology*. 2018;140:52–67. doi: 10.1159/000487936. ](https://pubmed.ncbi.nlm.nih.gov/29874661/) 



**---Ini adalah bagian akhir laporan---**
