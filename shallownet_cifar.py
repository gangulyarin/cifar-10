import argparse
from imutils import paths
from filedatasetloader import FileDatasetLoader
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import classification_report
from conv.shallownet import ShallowNet

ap = argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,help="Path to input dataset")
ap.add_argument('-s','--size',required=False,heimport oslp="Resize Dimension",default=32)
ap.add_argument('-e','--epochs',required=False,help="Number of Epochs",default=1)
args = vars(ap.parse_args())

train_path = args['dataset']+"/train"
test_path = args['dataset']+"/test"

imagePaths_train = list(paths.list_images(train_path))
imagePaths_test = list(paths.list_images(test_path))

ohe= OneHotEncoder()

size = int(args['size'])
epochs = int(args['epochs'])

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,rescale=1./255,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
traingen = datagen.flow_from_directory(train_path,target_size=(size,size),batch_size=32)
valgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,target_size=(size,size),batch_size=32)

model = ShallowNet.build(size,size,3,classes=10)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005),loss="categorical_crossentropy",metrics=['accuracy'])

H = model.fit_generator(traingen,epochs=epochs,validation_data=valgen,validation_steps=valgen.samples//32,steps_per_epoch=traingen.samples//32)

print("Evaluating...")
loader = FileDatasetLoader()
test = loader.load(imagePaths_test)
y_pred = model.predict_generator(valgen)
print(classification_report(ohe.fit_transform(test['Label'].values.reshape(-1,1)).toarray().argmax(axis=1),y_pred.argmax(axis=1)))
