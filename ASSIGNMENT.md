# Условия из архива CV_L2.zip

### Tema: Segmentarea imaginilor cu ajutorul rețelelor neuronale convoluționale

### Sarcini
1. Descărcați setul de date
    https://drive.google.com/drive/folders/1bltDZlNLoN8W2wPb5ZdZ0NNawBKTRLDT?usp=drive_link

2. Studiati setul de date descarcat.
3. Urmați pașii descriși în lucrare (mai jos de p.4).
4. Alegeti un alt set de date, spre ex. din Kaggle, legat de identificarea si segmentarea obiectelor din imagine sau va puteti crea unul nou si repetați pasii (bazat pe p.3) cu noul set de date.
**Pentru a crea un set de date pentru segmentarea semantică, vei urma pașii de bază pentru colectarea, preprocesarea și etichetarea datelor. Iată cum poți face acest lucru:**

    4.1. Colectarea imaginilor
      - Alege un domeniu specific pentru segmentare (de exemplu, documente, mașini, clădiri, animale etc.).
      - Fie că fotografiezi propriile imagini sau descarci un set de imagini din surse publice, ai grijă să ai un set de imagini suficient de variat.
      - Pentru a construi un set de date personal, vei avea nevoie de suficiente imagini care să reprezinte toate clasele de obiecte pe care vrei să le segmentezi.

    4.2. Etichetarea imaginilor (crearea măștilor)
   - Fiecare imagine din setul tău de date va avea o **mască** care arată care părți ale imaginii corespund anumitor clase (ex. clasa „cer”, „drum”, „mașină”).
   - Folosește un instrument de etichetare a imaginilor pentru a crea măști semantice. Iată câteva exemple de unelte utile:
     - **Labelbox**
     - **VGG Image Annotator (VIA)**
     - **LabelMe**
     - **CVAT**

   - În timpul etichetării, fiecare pixel din imagine va fi asociat cu o etichetă corespunzătoare unei clase (de exemplu, 0 pentru fundal, 1 pentru drum, 2 pentru mașină etc.).

    4.3. Preprocesarea imaginilor cu OpenCV
   - **Redimensionează imaginile**: Asigură-te că toate imaginile și măștile au aceleași dimensiuni (de ex., 256x256 pixeli).
   - Alte elemente de preprocesare pe care le crezi necesare
   - **Împărțirea datelor**: Împarte setul de date în:
     - *Set de antrenament* (80%)
     - *Set de validare* (10%) optional
     - *Set de testare* (10%)

    4.4 Salvarea și organizarea setului de date
   - **Organizează setul de date** într-o structură de directoare clară:
     ```
     Dataset/
       ├── images/
       │   ├── train/
       │   ├── val/
       │   └── test/
       └── masks/
           ├── train/
           ├── val/
           └── test/
     ```
   - Pune imaginile și măștile corespunzătoare în directoarele respective.

    4.5. **Antrenarea modelului**
   - Odată ce setul de date este gata, poți utiliza modelul U-Net sau alt model de segmentare semantică (YOLO) pentru antrenare, folosind datele personalizate.

## Основные параметры из notebook

- `conv_block`: две Conv2D 3×3, ReLU, same, he_normal; Dropout при необходимости; MaxPooling 2×2.
- `upsampling_block`: Conv2DTranspose 3×3, stride 2, same; concatenate со skip; две Conv2D.
- U-Net: 5 блоков encoder, 4 decoder; Dropout 0.3 в четвёртом и пятом encoder; последний без pooling.
- Вход 96×128×3, выход 23 класса, Adam и SparseCategoricalCrossentropy(from_logits=True).
- Обучение, график точности и сравнение настоящих и предсказанных масок. В примере предложено 40 эпох.
