import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel("C:\Users\ADMIN\Downloads\model.json");
const example = tf.fromPixels(webcamElement);  // for example

Xnew = np.asanyarray([[0,21,4,1,1,5,3,5,2,3,3,4,2,4,2,5,3]])


Xnew = preprocessing.StandardScaler().fit(Xnew).transform(Xnew)


ynew = model.predict(Xnew)
predict_classes=np.argmax(ynew,axis=1)

print("X=%s, Predicted=%s" % (Xnew[0], predict_classes[0]))

const prediction = model.predict(Xnew);

