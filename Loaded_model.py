

import numpy as np
import pandas as pd
import pickle 

# Loading saved model
loaded_model=pickle.load(open('C:/Users/LENOVO/Desktop/ML main/trained_model.sav','rb'))

input_data=(166205.0,-1.359807134,	-0.072781173,2.536346738,1.378155224,-0.33832077,0.462387778,0.239598554,0.098697901,0.3637869,0.090794172,-0.551599533,-0.617800856,-0.991389847,-0.311169354,1.468176972,-0.470400525,0.207971242,0.02579058,0.40399296,0.251412098,-0.018306778,0.277837576,-0.11047391,0.066928075,0.128539358,-0.189114844,0.133558377,-0.021053053,149.62)       
#changing the input_data as numpy array 
input_data_as_numpy_array = np.asarray(input_data)
#reshape the array as we are predicting
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
  print('The user is a Valid User')
else:
  print('The user is a Invalid User')  