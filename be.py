# importing the libraries
from fastapi import FastAPI, File, UploadFile, Form
from server_v2 import main
import os

# Instansiating FAST API  Server
app = FastAPI()
file_directory = lambda x:os.path.join(os.getcwd(),'docs',x)
os.makedirs(file_directory(''), exist_ok=True)
@app.post("/upload-pdf/") # end-point
async def upload_pdf(file: UploadFile = File(...),fname: str = Form(), query: str = Form()):
    # If the uploaded file is a PDF
    if fname == "None":
        if not file.filename.endswith(".pdf"):
            return {"Error": "Uploaded file is not a PDF"}
        with open(file_directory(file.filename), "wb+") as fp:
            fp.write(file.file.read())
        fname = file_directory(file.filename)
    response = main(fname, query=query)
    response["fname"] = fname
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
