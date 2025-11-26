from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np

m = load_model("model/digit_model.h5")
(x_train,y_train),(x_test,y_test)=mnist.load_data()
img = x_test[0].reshape(1,28,28,1)/255.0
pred = m.predict(img)
print("Predicted:", np.argmax(pred), "True:", y_test[0])
print("Top probs:", np.round(np.sort(pred[0])[-3:][::-1],4))
