from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
import json 
import os 

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

class TopicCreateRequest(BaseModel):
    name: str
    description: str

class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth: str = None 

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest, use_stored_emails: bool = False):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email, use_stored_emails = use_stored_emails)
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
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
    
@router.post("/topics")
async def add_topic(request: TopicCreateRequest): 
    file_path = "data/topic_keywords.json"
    
    try:
        with open(file_path, 'r') as f:
            topics = json.load(f)
            
        topics[request.name] = {"description": request.description}
        
        with open(file_path, 'w') as f:
            json.dump(topics, f, indent=4)
            
        return {"status": "success", "message": f"topic '{request.name}' added."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(0))
        
  
@router.post("/emails")
async def store_email(request: EmailStoreRequest): 
    file_path = "data/emails.json"
    
    try:
        with open(file_path, 'r') as f:
            stored_emails = json.load(f)
            
        stored_emails.append(request.dict())
        
        with open(file_path, 'w') as f:
            json.dump(stored_emails, f, indent=4)
            
        return {"status": "success", "message": "Email added to emails.json", "total_emails": len(stored_emails)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving email: {str(e)}")
        
        
        
        
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

