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
from urllib.parse import urlparse

size=32
epochs=1


def construct_model():
    model = ShallowNet.build(size,size,3,classes=10)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005),loss="categorical_crossentropy",metrics=['accuracy'])
    return model

def load_preprocess():

    args = {'dataset':'cifar_10/data'}
    train_path = args['dataset']+"/train"
    test_path = args['dataset']+"/test"
    
    return train_path,test_path 
    
    
    
def fit_model(**kwargs):
    ohe= OneHotEncoder()
    

    ti = kwargs['ti']
    loaded = ti.xcom_pull(task_ids='load_preprocess')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,rescale=1./255,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
    traingen = datagen.flow_from_directory(loaded[0],target_size=(size,size),batch_size=32)
    valgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(loaded[1],target_size=(size,size),batch_size=32)
    imagePaths_test = list(paths.list_images(loaded[1]))

    #mlflow.set_tracking_uri('http://localhost:5000')

    model = construct_model()

    with mlflow.start_run():
        H = model.fit_generator(traingen,epochs=epochs,validation_data=valgen,validation_steps=valgen.samples//32,steps_per_epoch=traingen.samples//32)

        print("Evaluating...")
        loader = FileDatasetLoader()
        test = loader.load(imagePaths_test)
        y_pred = model.predict_generator(valgen)
        #print(classification_report(ohe.fit_transform(test['Label'].values.reshape(-1,1)).toarray().argmax(axis=1),y_pred.argmax(axis=1)))
        mlflow.log_metric("Precision Score",precision_score(ohe.fit_transform(test['Label'].values.reshape(-1,1)).toarray().argmax(axis=1),y_pred.argmax(axis=1),average='micro'))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.keras.log_model(model, "model", registered_model_name="CIFAR10ShallowNet")
        else:
            mlflow.keras.log_model(model, "model")