from fastapi import FastAPI
import schemas
import pickle


app=FastAPI()

@app.get('/Pima_Diabetes')
def show():
    return 'Pima_Diabetes'

pickle_in=open('LR_classifier.pkl','rb')
pred=pickle.load(pickle_in)

mp=['Non Diabetic','Diabetic']
@app.post('/predict')
def predict(request:schemas.independent):
    res=pred.predict([[request.Pregnancies,request.Glucose,request.BloodPressure,request.SkinThickness,
    request.Insulin,request.BMI,request.DiabetesPedigreeFunction,request.Age]])

    return "The predicted class is : " + mp[int(res)]
    