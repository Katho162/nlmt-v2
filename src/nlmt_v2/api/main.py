from fastapi import FastAPI
from pydantic import BaseModel
from nlmt_v2.recommend import recommend as get_recommendations

app = FastAPI()

class RecommendationRequest(BaseModel):
    known_languages: list[str]
    top_k: int = 10

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    recommendations = get_recommendations(request.known_languages, request.top_k)
    return {"recommendations": recommendations}
