from fastapi import FastAPI
import main
import uvicorn
import joblib




    
app = FastAPI()
m = joblib.load('classifier.joblib')
gnb = joblib.load('gnb.joblib')
ran = joblib.load('randFor.joblib')
# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
#/{Cpu_Type}/{Cpu_core}/{Cpu_gen}/{Clockable}/{GPU}/{Gpu_gen}/{Gpu_type}
@app.get('/get_model/')
async def get_model(Cpu_Type : str , Cpu_core : int , Cpu_gen : str, Clockable : str, GPU : str,Gpu_gen : str,Gpu_type:str,price : int):
    return main.final_row(ran,Cpu_Type,Cpu_core,Cpu_gen,Clockable,GPU,Gpu_gen,Gpu_type,price)

# def get_name(Cpu_Type : str , Cpu_core : int , Cpu_gen : str, Clockable : str, GPU : str,Gpu_gen : str,Gpu_type:str):



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

