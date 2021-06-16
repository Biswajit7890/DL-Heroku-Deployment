

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class intel_image:
    def __init__(self,filename):
        self.filename =filename


    def predictionImage(self):
        # load model
        model = load_model('Intel_Image.h5')

         
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (32, 32))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)

        if(result== 0):
            prediction = 'Buildings'
            return [{ "image" : prediction}]
        elif(result==1): 
            prediction = 'Forest'
            return [{ "image" : prediction}]
        elif(result==2): 
            prediction = 'Glacier'
            return [{ "image" : prediction}]
        elif(result==3): 
            prediction = 'mountain'
            return [{ "image" : prediction}]
        elif(result==4): 
            prediction = 'sea'
            return [{ "image" : prediction}]
        else:
            prediction = 'street'
            return [{ "image" : prediction}]
        


