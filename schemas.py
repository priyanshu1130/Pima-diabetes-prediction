from pydantic import BaseModel

class independent(BaseModel):
    Pregnancies:float	
    Glucose:float
    BloodPressure:float	
    SkinThickness:float	
    Insulin:float	
    BMI:float	
    DiabetesPedigreeFunction:float
    Age:float