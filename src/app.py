from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.classifier import XGBoostClassifier, XGBoostClassifierSemantic
from typing import Optional

app = FastAPI()

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# classifier = XGBoostClassifier()
classifier_semantic = XGBoostClassifierSemantic()


@app.post("/classify_file")
async def classify_file_route(file: Optional[UploadFile] = File(None)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    filename = file.filename
    file_class = await classifier_semantic.predict(filename)

    return JSONResponse(content={"file_class": file_class})


# @app.post("/train_model")
# async def train_model_route():
#     """
#     Endpoint to train the model.

#     This function loads the training and testing data, trains the model using the
#     training data, evaluates the model using the testing data, and returns a JSON
#     response with the training results.
#     """
#     df_train, df_test = classifier.load_data()
#     scores = classifier.train_model(df_train, df_test)
#     response_content = {"message": "Model trained successfully"}
#     response_content.update(scores)
#     return JSONResponse(content=response_content)


@app.post("/train_model_with_openai")
async def train_model_semantic_route():
    """
    Endpoint to train the model.

    This function loads the training and testing data, trains the model using the
    training data, evaluates the model using the testing data, and returns a JSON
    response with the training results.
    """
    df_train, df_test = classifier_semantic.load_data()
    scores = classifier_semantic.train_model(df_train, df_test)
    response_content = {"message": "Model trained successfully"}
    response_content.update(scores)
    return JSONResponse(content=response_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
