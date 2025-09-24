import os
import json
import math
from typing import Dict, Any, List, Tuple
from app.features.factory import GENERATORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EmailClassifierModel:
    """Simple rule-based email classifier model"""
    
    def __init__(self):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())
    
    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)
    
    def predict(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using feature similarity"""
        scores = {}
        
        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        return max(scores, key=scores.get)
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate similarity score based on length difference"""
        # Get email embedding from features
        email_embedding = features.get("email_embeddings_average_embedding", 0.0)
        
        # Get topic description and create embedding (description length as embedding)
        topic_description = self.topic_data[topic]['description']
        topic_embedding = float(len(topic_description))
        
        # Calculate similarity based on inverse distance
        # Smaller distance = higher similarity
        distance = abs(email_embedding - topic_embedding)
        
        # Normalize to 0-1 range using exponential decay
        # e^(-distance/scale) gives values between 0 and 1
        scale = 50.0  # Adjust this to control how quickly similarity drops with distance
        similarity = math.exp(-distance / scale)
        
        return similarity
    
    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}

    def _email_corpus(self) -> tuple[list[str], list[str]]:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data"
        )
        emails_path = os.path.join(data_dir, "emails.json")

        if not os.path.exists(emails_path):
            return [], []

        with open(emails_path, "r") as f:
            items = json.load(f)

        texts: list[str] = []
        labels: list[str] = []

        for it in items:
            subj = (it.get("subject") or "").strip()
            body = (it.get("body") or "").strip()
            topic = (it.get("topic") or "").strip()
            text = f"Subject: {subj}\n\n{body}".strip()
            texts.append(text) 
            labels.append(topic)

        return texts, labels


    def predict_cosine_similarity(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using cosine similarity"""

        corpus_texts, corpus_labels = self._email_corpus()
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True)
        corpus_matrix = tfidf.fit_transform(corpus_texts)
        email_vec = tfidf.transform(features)


        sims = cosine_similarity(email_vec, corpus_matrix).ravel()

        best_idx = int(np.argmax(sims))
        return corpus_labels[best_idx]