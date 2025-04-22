from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from reccomender import recommend_assessments

app = FastAPI()

class QueryInput(BaseModel):
    query: str
    max_duration: int | None = None

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
async def recommend(input: QueryInput):
    try:
        recommendations = recommend_assessments(input.query, input.max_duration)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))