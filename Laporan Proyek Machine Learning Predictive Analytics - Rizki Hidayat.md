# Laporan Proyek Machine Learning Predictive Analytics - Rizki Hidayat

## Domain Proyek

Penyakit jantung koroner adalah salah satu penyebab utama kematian di seluruh dunia. Memprediksi penyakit jantung adalah salah satu tugas yang paling menantang di bidang analisis data klinis. Machine learning (ML) berguna dalam bantuan diagnostik dalam hal pengambilan keputusan dan prediksi berdasarkan data yang dihasilkan oleh sektor perawatan kesehatan secara global. Kami juga telah merasakan teknik ML yang digunakan dalam bidang medis untuk prediksi penyakit. Dalam hal ini, banyak studi penelitian telah ditunjukkan pada prediksi penyakit jantung menggunakan pengklasifikasi ML.



**Rubrik/Kriteria Tambahan**:
Menurut Organisasi Kesehatan Dunia [1], CVD adalah penyebab kematian terbesar di dunia, yang mengakibatkan kematian sekitar 17,9 juta orang setiap tahunnya. 

Industri perawatan kesehatan menghasilkan banyak data mengenai pasien, penyakit, dan diagnosis, tetapi tidak dianalisis dengan benar, sehingga tidak memiliki dampak yang sama seperti yang seharusnya pada kesehatan pasien[1]

CVD meliputi arteri koroner, penyakit jantung rematik, penyakit pembuluh darah, dan berbagai masalah jantung dan pembuluh darah. Empat dari setiap lima kematian akibat CVD disebabkan oleh stroke atau serangan jantung. Di antara total kematian, sepertiganya terjadi pada orang yang berusia di bawah 70 tahun [2]

Jenis kelamin, merokok, usia, riwayat keluarga, pola makan yang buruk, kolesterol, kurangnya aktivitas fisik, tekanan darah tinggi, kelebihan berat badan, dan penggunaan alkohol adalah pengaruh risiko utama penyakit jantung. Penyakit jantung juga disebabkan oleh faktor risiko keturunan seperti diabetes dan tekanan darah tinggi [3].

 Kelelahan, jantung berdebar, berkeringat, nyeri punggung, nyeri dada, nyeri bahu dan lengan, sesak napas, dan kelemahan secara keseluruhan adalah gejala yang paling umum. Tanda yang paling sering muncul dari kurangnya aliran darah ke jantung adalah nyeri dada. Dalam istilah medis, nyeri dada jenis ini dikenal sebagai Angina [4]. Ada beberapa pemeriksaan yang tersedia untuk membantu mendiagnosis penyakit ini, seperti sinar-X, pemindaian MRI, dan angiografi. Namun, ada kalanya terjadi kekurangan sumber daya dalam keadaan darurat karena tidak tersedianya peralatan medis. Pada penyakit kardiovaskular, waktu sama pentingnya dengan setiap momen dalam mendiagnosis dan mengobati penyakit [4].
 
 Sehingga perlu dilakukan upaya untuk mengolah data kesehatan terkait penyakit jantung ini, agar dapat dilakukan prediksi dini untuk menentukan apakah pasien memiliki resiko penyakit jantung atau tidak.
 
 Perkembangan pesat teknologi AI dan Machine Learning membuat prediksi dini tersebut dapat dilakukan dengan menggunakan Model Machine Learning.
 
 Pengembangan Model Machine Learning untuk melakukan prediksi resiko penyakit jantung ini telah banyak dilakukan oleh peneliti
  
  Format Referensi: 
  1. [World Health Organization Cardiovascular Diseases (CVDs)](https://www.who.int/health-topics/cardiovascular-diseases/#tab=tab_1) 
  2. [AFRO.WHO Cardiovascular Diseases (CVDs)](https://www.afro.who.int/health-topics/cardiovascular-diseases) 
  3. [Heart.org Why High Blood Pressure is a silent killer]( https://www.heart.org/en/health-topics/high-blood-pressure/why-high-blood-pressure-is-a-silent-killer/know-your-risk-factors-for-high-blood-pressure) 
  4. [Balla C., Pavasini R., Ferrari R. Treatment of Angina: Where Are We? *Cardiology*. 2018;140:52–67. doi: 10.1159/000487936. ](https://pubmed.ncbi.nlm.nih.gov/29874661/) 
  5. [Rumsfeld J.S., Joynt K.E., Maddox T.M. Big data analytics to improve cardiovascular care: Promise and challenges. Nat. Rev. Cardiol. 2016;13:350–359. doi: 10.1038/nrcardio.2016.42.](https://pubmed.ncbi.nlm.nih.gov/27009423/)
  6. [Maryam A., Mahmoud Q., Mohammad H. Machine Learning Classification Techniques for Heart Disease Prediction: A Review. Int. J. Eng. Technol. 2018;7:5373–5379. doi: 10.14419/ijet.v7i4.28646.](https://scholar.google.com/scholar_lookup?journal=Int.+J.+Eng.+Technol.&title=Machine+Learning+Classification+Techniques+for+Heart+Disease+Prediction:+A+Review&author=A.+Maryam&author=Q.+Mahmoud&author=H.+Mohammad&volume=7&publication_year=2018&pages=5373-5379&doi=10.14419/ijet.v7i4.28646&)

## Business Understanding
Berdasarkan informasi dari WHO, terdapat banyak faktor yang dapat mempengaruhi risiko seseorang terkena penyakit jantung atau tidak.

Untuk itu penting untuk diketahui faktor apa saja kah yang sangat memperngaruhi risiko penyakit jantung tersebut.

Sehingga kedepannya, resiko penyakit jantung tersebut dapat dideteksi secara dini, dan dapat menyelamatkan nyawa pasien.

### Problem Statements

Berdasarkan kondisi tersebut maka perlu dikembangkan sistem prediksi untuk dapat menjawab permasalahan tersebut:
- Dari serangkaian fitur yang ada, fitur apa yang peling berpengaruh terhadap risiko penyakit jantung?
- Apakah resiko penyakit jantung pasien dapat dideteksi secara dini?

### Goals

Untuk menjawab pertanyaan tersebut akan dibuatkan Model Predictive Analysis untuk malasah Klasifikasi dengan tujuan sebagai berikut :
- Mengetahui fitur yang paling berpengaruh terhadap risiko penyakit jantung
- Membuat model machine learning yang dapat mendeteksi risiko penyakit jantung dengan akurasi terbaik

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution statements

Untuk dapat menyelesaikan masalah tersebut saya akan melakukan modeling dengan menggunakan 5 base model dan melakukan Hyperparameter tuning degan bantuan fungsi GridSearchCV.

5 model yang saya gunakan antara lain:
1. Random Forest Calssifier
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Ada Boost, and
5. XG Boost

## Data Understanding
Dataset yang digunakan untuk Tugas kali ini adalah UCI Heart Disease Dataset dari Kaggle.

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


### Variabel-variabel UCI Heart Disesase dataset adalah sebagai berikut:
1. id : Unique id for each patient
2. age : Age of the patient in years
3. sex : Gender (Male/Female)
4. dataset : Place of study (Cleveland, Hungary, Switzerland, VA Long Beach)
5. cp : chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)
6. trestbps : resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
7. chol : serum cholesterol in mg/dl
8. fbs : if fasting blood sugar > 120 mg/dl (True/ Flase)
9. restecg : resting electrocardiographic results (normal, stt abnormality, lv hypertrophy)
10. thalach : maximum heart rate achieved
11. exang : exercise-induced angina (True/ False)
12. oldpeak : ST depression induced by exercise relative to rest
13. slope : the slope of the peak exercise ST segment (upsloping, flat, downsloping)
14. ca : number of major vessels (0-3) colored by fluoroscopy
15. thal : Thalassemia (normal; fixed defect; reversible defect)
16. num: the predicted attribute (0: no heart disease, 1,2,3,4: stage of heart disease)

**Rubrik/Kriteria Tambahan (Opsional)**:
Untuk lebih memahami dataset yang digunakan, saya melakukan teknik Exploratory Data Analysis pada data

### Exploratory Data Analysis
Tahap yang pertama adalah melakukan Univariate Analysis untuk fitur kategori dan fitur numerik.

Setelah melakuakn Univariate Analysis, selanjutnya melakukan Bivariate Analysis dengan menbandingkan fitur kategori dan fitur numerik dengan target

#### Univariate Analysis - Fitur Kategori
Analisis ini dilakukan dengan menggunakan bar plot untuk fitur kategori.
Fitur - fitur kategori pada dataset ini adalah : ['sex','cp','fbs','restecg','exang','slope','ca','thal']

##### Sex (Jenis kelamin)
Barplot untuk fitur 'sex' adalah :
![Barplot_sex]('https://drive.google.com/uc?export=download&id=1kKmgybz_wzX6Jj0G_tnpY014NVVureXr')




## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

- Import a HTML file and watch it magically convert to Markdown
- Drag and drop images (requires your Dropbox account be linked)
- Import and save files from GitHub, Dropbox, Google Drive and One Drive
- Drag and drop markdown and HTML files into Dillinger
- Export documents as Markdown, HTML and PDF

Markdown is a lightweight markup language based on the formatting conventions
that people naturally use in email.
As [John Gruber] writes on the [Markdown site][df1]

> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.

This text you see here is *actually- written in Markdown! To get a feel
for Markdown's syntax, type some text into the left window and
watch the results in the right.

## Tech

Dillinger uses a number of open source projects to work properly:

- [AngularJS] - HTML enhanced for web apps!
- [Ace Editor] - awesome web-based text editor
- [markdown-it] - Markdown parser done right. Fast and easy to extend.
- [Twitter Bootstrap] - great UI boilerplate for modern web apps
- [node.js] - evented I/O for the backend
- [Express] - fast node.js network app framework [@tjholowaychuk]
- [Gulp] - the streaming build system
- [Breakdance](https://breakdance.github.io/breakdance/) - HTML
to Markdown converter
- [jQuery] - duh

And of course Dillinger itself is open source with a [public repository][dill]
 on GitHub.

## Installation

Dillinger requires [Node.js](https://nodejs.org/) v10+ to run.

Install the dependencies and devDependencies and start the server.

```sh
cd dillinger
npm i
node app
```

For production environments...

```sh
npm install --production
NODE_ENV=production node app
```

## Plugins

Dillinger is currently extended with the following plugins.
Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:

```sh
node app
```

Second Tab:

```sh
gulp watch
```

(optional) Third:

```sh
karma test
```

#### Building for source

For production release:

```sh
gulp build --prod
```

Generating pre-built zip archives for distribution:

```sh
gulp build dist --prod
```

## Docker

Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

```sh
cd dillinger
docker build -t <youruser>/dillinger:${package.json.version} .
```

This will create the dillinger image and pull in the necessary dependencies.
Be sure to swap out `${package.json.version}` with the actual
version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on
your host. In this example, we simply map port 8000 of the host to
port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart=always --cap-add=SYS_ADMIN --name=dillinger <youruser>/dillinger:${package.json.version}
```

> Note: `--capt-add=SYS-ADMIN` is required for PDF rendering.

Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## License

MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
