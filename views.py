"""
Routes and views for the flask application.
"""

from datetime import datetime
from FlaskWebProject2 import app
from werkzeug import secure_filename
import pandas as pd
import numpy as np
import math
import random
from flask import render_template,request, url_for, redirect, session
import os

from sklearn.preprocessing import normalize

app.secret_key = os.urandom(12)

@app.route('/')
@app.route('/dashboard')
def home():
    """Renders the home page."""
    #return HTMLPage1()
    return render_template(
        'knnsmote.html',
        title='Login Page'
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != '123':
            error = 'Username or Password Invalid! Please Try Again'
        else:
            session['logged_in'] = True
            return redirect(url_for('afterlogin'))
    return render_template('index.html', error=error)

@app.route("/afterlogin")
def afterlogin():
    if not session.get('logged_in') :
        return redirect(url_for('login'))
    else:
        return render_template('dashboard_1.html')

@app.route('/profile')
def profile():
    if not session.get('logged_in') :
        return redirect(url_for('login'))
    else:
        return render_template('user1.html')

@app.route('/dashboard_1')
def dashboard_1():
    if not session.get('logged_in') :
        return redirect(url_for('login'))
    else:
        return render_template('dashboard_1.html')

@app.route('/knnsmote')
def knnsmote():
    if not session.get('logged_in') :
        return redirect(url_for('login'))
    else:
        return render_template('knnsmote.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect('knnsmote.html')
        # KNN-SMOTE
        f = request.files['file']
        dtest = bacaFile(f)
        test_class = dtest[6]
        dtrain = bacaFile('tr78.csv')
        klsmin = cariKelasMinor(dtrain[6])
        datamin = dataKelasMinor(klsmin, dtrain[6], dtrain[5])
        datasmote = smote(200, 9, klsmin[1], klsmin, datamin)
        combine = combineSintetis(dtrain[5], datasmote, dtrain[6])
        jrk = cariJarak(combine, dtest[5])
        urut = urutkanJarak(jrk, dtest[5])
        tetangga = cariTetangga(urut, 3)
        kls = menentukanKelas(tetangga)
        tampil = tampilkan(dtest, kls)
        akurasi = hitungAkurasi(kls, tetangga, dtest[6])
        confmatriks = hitungConfMatriks(kls, dtest[6], tetangga)
        text = ['Nomor Akun', 'Date', 'Nama Akun', 'Deskripsi', 'Data Kredit', 'Kelas Awal', 'Hasil Klasifikasi']

        # Grafik KNN-SMOTE
        global jumlahPerKelas
        jumlahPerKelas = grafik(dtest[6], kls)
        if request.form['submit_button'] == 'upload':
            return render_template("knnsmote.html", da = text, column = tampil)
                            
    elif request.method == 'GET':
        return rerender_template('knnsmote.html')

# Method untuk melakukan pembacaan file data test
def bacaFile(nama):
    data = pd.read_csv(nama)
    no_akun = data['No Akun'].values.tolist()
    date = data['Date'].values.tolist()
    nama_akun = data['Nama Akun'].values.tolist()
    deskripsi = data['Description'].values.tolist()
    data_credit = data[['Credit']].values.tolist()
    data_class = data['Class'].values.tolist()
    data_norm = normalize(data_credit, axis=0, norm='max')
    return (no_akun, date, nama_akun, deskripsi, data_credit, data_norm, data_class)

# Menentukan kelas minor dengan menghitung jumlah kemunculan masing-masing kelas
def cariKelasMinor(train_class):
    kelas = [0,0]
    for i in range(0,len(train_class)):
        if str(train_class[i])=='1':
            kelas[0] +=1
        elif str(train_class[i])=='2':
            kelas[1] +=1
    if kelas[0] < kelas[1]:
        return 1, kelas[0]
    else:
        return 2, kelas[1]

# Mengambil data yang termasuk dalam kelas minoritas
def dataKelasMinor(kelasMinor, train_class, trainc):
    dataMinor = [0 for x in range(0,kelasMinor[1])]
    j = 0
    for i in range(0, len(train_class)):
        if str(train_class[i]) == str(kelasMinor[0]):
            dataMinor[j] = float(trainc[i])
            j += 1
    return dataMinor

# SMOTE (Membuat data sintetis untuk ditambahkan pada data learning)
def smote(N,k,T, kelasMinor, dataMinor):
    #    Menentukan jumlah kelipatan data yang akan dibuat
    dataDiulang = int(T)
    if N<100:
        dataDiulang =int((N/100)*int(T))
        N =100
    
    N = int(N/100)
    
#    Mencari jarak euclidean, 
    euclideanD = [[0 for x in range(0,kelasMinor[1])] for y in range(kelasMinor[1])]
    for i in range(0, len(dataMinor)):
        for j in range(0, len(dataMinor)):
            euclideanD[i][j] = math.sqrt(math.pow(dataMinor[i]-dataMinor[j],2))
    
#    Mengurutkan jarak euclidean dari yang terkecil ke terbesar
    jarakUrut = [[0 for x in range(0,kelasMinor[1])] for y in range(kelasMinor[1])]
    for i in range(0, kelasMinor[1]):
        jarakUrut[i] = sorted(euclideanD[i])

#    Mencari tetangga
    tetangga = [[0 for x in range(k)] for y in range(len(jarakUrut))]
    for i in range(len(jarakUrut)):
        newindex = 0
        for j in range(k):
            if i == 1:
                continue
#            else :
            tetangga[newindex][j] = jarakUrut[i][j+1]
            newindex += 1
    
#    Membuat data sintetis
    sintetis = [[0 for x in range(0,2)] for y in range(dataDiulang*N)]
    newindex = 0
    while N != 0:
        nn = random.randint(0, (k-1))
        print(nn)
        for i in range (dataDiulang):
            rand = random.random()
            sintetis[newindex] = [dataMinor[i]+(tetangga[i][nn] - dataMinor[i])*rand,kelasMinor[0]]
#            print(tetangga[i][nn])
#            print("random:", rand)
            newindex += 1
        N -= 1
    return sintetis

# Menggabungkan data sintetis hasil SMOTE ke data training
def combineSintetis(trainc, dataSintetis, train_class):
    dataTrain = [[0 for x in range(0,2)] for y in range(len(trainc)+len(dataSintetis))]
    indexBaru = 0
    for i in range(len(trainc)):
        dataTrain[i] = [float(trainc[i]),train_class[i]]
        indexBaru += 1
    for j in range(len(dataSintetis)):
#        print(dataSintetis[j])
        dataTrain[indexBaru] = [dataSintetis[j][0],dataSintetis[j][1]]
        indexBaru += 1
    return dataTrain

# Mencari jarak euclidean dari setiap data test ke data training
def cariJarak (trainCombine, test):
    jarak = [[0 for x in range(len(trainCombine))] for y in range(len(test))] 
    for i in range(len(trainCombine)):
        for j in range(len(test)):
            jarak[j][i] = [math.sqrt(math.pow(trainCombine[i][0] - test[j],2)),trainCombine[i][1]]
    return jarak

# Mengurutkan jarak eucliedean data test dari yang terkecil ke terbesar
def urutkanJarak (jarak, testc):
    arr = jarak
    for i in range(len(testc)):
        arr[i] = sorted(jarak[i], key=lambda x: x[0])
    return arr

# Menentukan ketetanggaan dari data test
def cariTetangga(jarakUrut,k):
    tetangga = [[0 for x in range(k)] for y in range(len(jarakUrut))]
    for i in range(len(jarakUrut)):
        for j in range(k):
            tetangga[i][j] = jarakUrut[i][j]
    return tetangga

# Menentukan kelas data test berdasarkan mayoritas kelas dari tetangga terdekatnya
def menentukanKelas(tetangga):
    arr = [[0 for x in range(0,2)] for y in range(len(tetangga))]
    for i in range(len(arr)):
        for j in range(len(tetangga[0])):
            if str(tetangga[i][j][1]) == '1':
                arr[i][0] +=1
            elif str(tetangga [i][j][1]) == '2':
                arr[i][1] +=1
    kelas = [0 for x in range(0,len(tetangga))]
    for i in range(len(arr)):
        if arr[i][0] >arr[i][1]:
            kelas[i] = '1'
        else:
            kelas[i] = '2'
    return kelas

def hitungAkurasi(kelas, tetangga, test_class):
    benar = 0
    for i in range(0, len(tetangga)):
        if str(test_class[i])==kelas[i]:
            benar +=1
#        print(str(test_class[i])+" " +kelas[i])
    akurasi =benar/len(kelas)
    print(akurasi)
    return akurasi

def hitungConfMatriks(kelas, test_class, tetangga):
    cm = [0 for i in range (0,4)]
    for i in range(0, len(tetangga)):
        if str(test_class[i])==kelas[i] and str(test_class[i])=='1':
            cm[0] +=1
        elif str(test_class[i])==kelas[i] and str(test_class[i])=='2':
            cm[3] +=1
        elif str(test_class[i])=='1' and kelas[i]=='2':
            cm[1] +=1
        else:
            cm[2] +=1
#         print(str(test_class[i])+" " +kelas[i])
    presisi =cm[0]/(cm[0]+cm[2])
    recall = cm[0]/(cm[0]+cm[1])
    spesifikasi = cm[3]/(cm[3]+cm[2])
    g_mean = math.sqrt(abs(recall-spesifikasi))
    f_measure = (2*recall*presisi)/(recall+presisi)
    print(presisi, recall, spesifikasi, g_mean, f_measure)
    return presisi, recall, spesifikasi, g_mean, f_measure

# Menentukan data yang akan ditampilkan pada tabel hasil klasifikasi
def tampilkan(data, kelas):
    dataTampil =  [0 for i in range(len(data[0]))]
    for i in range (0, len(dataTampil)):
        dataTampil[i] = [data[0][i],data[1][i],data[2][i],data[3][i],data[4][i][0],data[6][i], kelas[i]]

    return dataTampil

# Menghitung jumlah persebaran data hasil klasifikasi untuk ditampilkan dalam bentuk grafik
def grafik(data, kelas):
    jumlahPerKelas = [[0 for i in range(0,2)] for j in range(0,2)]
    for i in range(0,2):
        for j in range(0,2):
            if i == 0:
                jumlahPerKelas[i][j] = data.count(j+1)
            else:
                jumlahPerKelas[i][j] = kelas.count(str(j+1))
                
    return jumlahPerKelas

# Mengembalikan data yang akan ditampilkan pada grafik hasil klasifikasi KNN-SMOTE
@app.route('/grafik_knn')
def tampil_grafik():
   tampil = jumlahPerKelas
   label = ['Kelas Awal', 'Kelas hasil Klasifikasi']
   kelas = ['1','2']
   return render_template('grafik_knn.html', de=label, ga=kelas, tampil = tampil)

@app.route('/benford', methods = ['GET', 'POST'])
def benford():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect('tables4.html')
        f = request.files['file']
        # Benford
        databenford = bacaBenford(f)
        datastring = dataStr(databenford)
        digit = ambilDigit(datastring)
        munculdigit = hitungKemunculanDigit(digit)
        realbenford = hitungRealBenford(munculdigit)
        nilaibenfrd = nilaiBenford()
        selisih = hitungSelisihBenford(nilaibenfrd, realbenford)
        urut = urutkanTransaksi(selisih)
        tampilurut = tampilkanUrutan(urut, digit, databenford)
        debit = tampilurut[0]
        kredit = tampilurut[1]
        text = ['Date', 'Deskripsi', 'Data Kredit']
        text2 = ['Date', 'Deskripsi', 'Data Debit']
        
        # grafik Benford
        global dataGrafik
        dataGrafik = grafikBenford(realbenford, nilaibenfrd)
        return render_template('tables4.html', df = text, df2 = text2, dt = debit, kt = kredit, tuple = tuple, type = type)

    else:
        if not session.get('logged_in') :
            return redirect(url_for('login'))
        else:
            return render_template('tables4.html')

# BENFORD LAW

# Melakukan Pembacaan data
def bacaBenford(nama):
    data = pd.read_excel(nama)
    data_credit = data['Credit'].values.tolist()
    data_debit = data['Debit'].values.tolist()
    data_date = data['Date'].dt.strftime('%d/%m/%Y')
    data_description = data['Description'].values.tolist()
    return [data_credit,data_debit,data_date, data_description]

# Mengubah data kredit dan data debit menjadi bertipe data String
def dataStr(data):
    for i in range(len(data[0])):
        data[0][i] = str(data[0][i])
        data[1][i] = str(data[1][i])
    return data

# Mengambil digit pertama untuk setiap data kredit dan data debit
def ambilDigit(data):
    dataDigit = [[0 for i in range(0,len(data[0]))] for j in range(0,2)]
    for i in range(0, len(dataDigit[0])):
        for j in range(0,2):
            dataDigit[j][i] = data[j][i][0]
    return dataDigit

# Menghitung kemunculan digit 0-9 pada data kredit dan data debit
def hitungKemunculanDigit(data):
    digit = [i for i in range (1,10)]
    jumlahDigit = [[0 for i in range(0,9)] for j in range(0,2)]
    for y in range(0, 2):
        for i in range(0, len(data[0])):
            for j in range (0,9):          
                if data[y][i] == str(digit[j]):
                    jumlahDigit[y][j] += 1
    return jumlahDigit

# Menghitung nilai Benford berdasarkan kondisi riil data kredit dan data debit
def hitungRealBenford(data):
    nilaiBenford = [[0 for i in range(0,9)] for j in range(0,2)]
    jumlah = [0,0]
    for i in range(0,9):
        for j in range(0,2):
            jumlah[j] += data[j][i]
        
    for i in range(0,9):
        for j in range(0,2):
            nilaiBenford [j][i] = data[j][i]/jumlah[j]
    
    return nilaiBenford

# Menghitung koofesien nilai Benford untuk digit 0-9
def nilaiBenford():
    digit = [i for i in range (1,10)]
    benford = [0 for i in range(0, 9)]
    
    for i in range (0,9):
        benford[i] = np.log10(1+(1/digit[i]))
    
    return benford

# Menghitung selisih nilai riil Benford dengan nilai pada koefisien Benford
def hitungSelisihBenford(benford, realBenford):
    selisihBenford = [[0 for i in range(0, 9)] for j in range(0,2)]
    
    for i in range(0,9):
        for j in range(0,2):
            selisihBenford[j][i] = abs(benford[i]-realBenford[j][i])
    return selisihBenford

# Mengurutkan selisih nilai Benford dari yang terbesar ke terkecil (terbesar=transaksi perlu dicurigai, terkecil=tingkat kecurigaan kecil)
def urutkanTransaksi(data):
    dataUrut = [[[0 for i in range(0, 2)] for j in range(0,9)] for k in range(0,2)]
    digit = [i for i in range (1,10)]
    for i in range (0,2):
        for j in range(0,9):
            dataUrut [i][j] = data[i][j], digit[j]

    for i in range(0, 2):
        dataUrut[i] = sorted(dataUrut[i], key=lambda x: x[0], reverse=True)
    return dataUrut

# Membuat format data yang akan ditampilkan pada tabel
def tampilkanUrutan(dataUrut, dataDigit, data):
    tampilUrut = [[0 for j in range(0, len(data[0]))] for i in range(0,2)]
    for i in range(0,2):
        newindex = 0
        for j in range(0,9):
            for k in range(0, len(data[0])):
                if dataDigit[i][k] == str(dataUrut[i][j][1]) and data[i][k]:
                    tampilUrut[i][newindex]= data[2][k], data[3][k], data[i][k]
                    newindex += 1
    return tampilUrut

# Mengembalikan data yang akan ditampilkan pada grafik Benford Law
def grafikBenford(data, benford):
    dataGrafik = [0 for j in range(0,3)]
    dataGrafik[0] = data[0]
    dataGrafik[1] = benford
    dataGrafik[2] = data[1]
    return dataGrafik

@app.route('/HTMLPage1')
def HTMLPage1():
    c = ['Kredit','Benford','Debit']
    e = [i for i in range(1,10)]
    d = dataGrafik

    return render_template('HTMLPage1.html', de=c, ga=d, ha=e)

@app.route('/y')
def y():
    c = ['Kredit','Benford','Debit']
    e = [i for i in range(1,10)]
    d = dataGrafik

    return render_template('y.html', de=c, ga=d, ha=e)

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()

if __name__ == '__main__':
    app.run(debug = True)