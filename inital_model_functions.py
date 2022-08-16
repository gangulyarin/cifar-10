import argparse
from imutils import paths
from filedatasetloader import FileDatasetLoader
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import classification_report
from conv.shallownet import ShallowNet
import mlflow
import mlflow.keras
from sklearn.metrics import precision_score

size=32
epochs=1

def load_preprocess():
    args = {'dataset':'cifar-10/data'}
    train_path = args['dataset']+"/train"
    test_path = args['dataset']+"/test"

    imagePaths_train = list(paths.list_images(train_path))
    imagePaths_test = list(paths.list_images(test_path))

    ohe= OneHotEncoder()
    
    model = ShallowNet.build(size,size,3,classes=10)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005),loss="categorical_crossentropy",metrics=['accuracy'])

def fit_model():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,rescale=1./255,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
    traingen = datagen.flow_from_directory(train_path,target_size=(size,size),batch_size=32)
    valgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,target_size=(size,size),batch_size=32)

    mlflow.set_tracking_uri('http://mlflow:5000')

    with mlflow.start_run():
        H = model.fit_generator(traingen,epochs=epochs,validation_data=valgen,validation_steps=valgen.samples//32,steps_per_epoch=traingen.samples//32)

        print("Evaluating...")
        loader = FileDatasetLoader()
        test = loader.load(imagePaths_test)
        y_pred = model.predict_generator(valgen)
        #print(classification_report(ohe.fit_transform(test['Label'].values.reshape(-1,1)).toarray().argmax(axis=1),y_pred.argmax(axis=1)))
        mlflow.log_metric("Precision Score",precision_score(ohe.fit_transform(test['Label'].values.reshape(-1,1)).toarray().argmax(axis=1),y_pred.argmax(axis=1)))