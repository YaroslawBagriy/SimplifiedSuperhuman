from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
from app.features.factory import FeatureGeneratorFactory
from pathlib import Path
import json
from typing import Optional
from enum import Enum

router = APIRouter()

class Strategy(str, Enum):
    TOPIC = "topic"
    SIMILARITY = "similarity"

class EmailRequest(BaseModel):
    subject: str
    body: str
    strategy: Strategy = Strategy.TOPIC

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    strategy: Strategy = Strategy.TOPIC

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

class Topic(BaseModel):
    topic: str
    description: str

@router.post("/emails")
async def emails(email: EmailWithTopicRequest):
    """Post new email"""
    try:
        file_path = Path.cwd() / "data" / "emails.json"
        emails_store = json.loads(file_path.read_text(encoding="utf-8"))

        # Convert to dict
        entry = email.model_dump(exclude_none=True, by_alias=True)
        emails_store.append(entry)

        file_path.write_text(json.dumps(emails_store, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        # if the topic is set to topic, then classify emails by topic
        # else classify emails by cosine similarity
        result =  inference_service.classify_email(email) if request.strategy is Strategy.TOPIC else inference_service.classify_email_similarity(email)
        return EmailClassificationResponse(
                predicted_topic=result["predicted_topic"],
                topic_scores=result["topic_scores"],
                features=result["features"],
                available_topics=result["available_topics"]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/topics")
async def topics(topic: Topic):
    """Post new email topic"""
    try:
        # Add new topic to topic_keywords store
        file_path = Path.cwd() / "data" / "topic_keywords.json"
        topic_keywords = json.loads(file_path.read_text(encoding="utf-8"))

        key = topic.topic
        # Check if the topic already exists
        # If it does, return false and don't add the topic
        if key in topic_keywords:
            return False

        # store as an object with a description as a field
        topic_keywords[key] = {"description": topic.description}

        file_path.write_text(json.dumps(topic_keywords, ensure_ascii=False, indent=2), encoding="utf-8")

        # Return True if topic was added successfully
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

