# Master Thesis
MAGISTER PENGINDERAAN JAUH
FAKULTAS GEOGRAFI
UNIVERSITAS GADJAH MADA

Repositori ini berisi kode yang digunakan dalam tesis magister saya, berjudul "ESTIMASI CURAH HUJAN DARI DATA MULTIKANAL HIMAWARI-8/9 MENGGUNAKAN MACHINE LEARNING". Penelitian ini bertujuan untuk mengembangkan dan mengevaluasi model pembelajaran mesin untuk estimasi curah hujan menggunakan data satelit.

├── notebooks
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
│   └── ...
├── src
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── ...
├── results
│   ├── figures
│   │   └── [grafik_estimasi.png]
│   ├── tables
│   │   └── [hasil_evaluasi.xlsx]
│   └── ...
├── README.md
└── requirements.txt

python src/data_preprocessing.py

for a in kluster:
    #baca data
    df=pd.read_csv(dir2+a)

    #clean missing value
    dfclean=df.dropna().reset_index(drop=True)

    #clean rr
    dfclean2=dfclean[dfclean['rr']<=30]

    # clean outlier satelit
    ## filter IR
    df7=dfclean2[dfclean2['I4']>=161.96]
    df8=df7[df7['WV']>=161.96]
    df9=df8[df8['W2']>=161.96]
    df10=df9[df9['W3']>=161.96]
    df11=df10[df10['MI']>=161.96]
    df12=df11[df11['O3']>=161.96]
    df13=df12[df12['IR']>=161.96]
    df14=df13[df13['L2']>=161.96]
    df15=df14[df14['I2']>=161.96]
    df16=df15[df15['CO']>=161.96]

    df17=df16[df16['I4']<996921]
    df18=df17[df17['WV']<996921]
    df19=df18[df18['W2']<996921]
    df20=df19[df19['W3']<996921]
    df21=df20[df20['MI']<996921]
    df22=df21[df21['O3']<996921]
    df23=df22[df22['IR']<996921]
    df24=df23[df23['L2']<996921]
    df25=df24[df24['I2']<996921]
    dfclean3=df25[df25['CO']<996921]


    ## filter VNIR
    nite=dfclean3[dfclean3['zenith']>=70]
    day=dfclean3[dfclean3['zenith']<70]
    #print(len(day),len(nite),(len(day)+len(nite)))

    df2=day[day['V1'] >= 0]
    df3=df2[df2['V2'] >= 0]
    df4=df3[df3['VS'] >= 0]
    df5=df4[df4['N1'] >= 0]
    df6=df5[df5['N2'] >= 0]
    df7=df6[df6['N3'] >= 0]

    df8=df7[df7['V1'] <= 1]
    df9=df8[df8['V2'] <= 1]
    df10=df9[df9['V2'] <= 1]
    df11=df10[df10['V2'] <= 1]
    df12=df11[df11['V2'] <= 1]
    df13=df12[df12['V2'] <= 1]
    dfclean4 = pd.concat([df13, nite])

    dfclean4.to_csv(dir2+'4.clean_'+a[2:])
    print(a[2:],len(df), len(dfclean), len(dfclean2), len(dfclean3), len(dfclean4))
