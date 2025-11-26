#!/usr/bin/env python3
"""
Quick test script to verify Ollama integration works
"""
import os
import json
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

# Test configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:latest')

def test_ollama_connection():
    """Test basic Ollama connectivity"""
    try:
        model = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0,
            num_predict=1500,
        )
        
        # Simple test prompt
        test_prompt = PromptTemplate(
            template="You are a helpful assistant. Please respond with a simple JSON object containing 'status': 'success' and 'message': 'Ollama is working properly'",
            input_variables=[]
        )
        
        chain = test_prompt | model
        result = chain.invoke({})
        
        print("✅ Ollama connection successful!")
        print(f"Model: {OLLAMA_MODEL}")
        print(f"Response: {result.content}")
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis with structured output"""
    try:
        model = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0,
            num_predict=1500,
        )
        
        # Test with sample survey responses
        sample_responses = """
        Question 1: I really enjoy working here. The team is supportive and I feel valued.
        Question 2: 4 (on a scale of 1-5, where 5 is very happy)
        Question 3: The benefits package is comprehensive and fair.
        Question 4: I would like more flexible working hours.
        Question 5: The onboarding process was thorough and helpful.
        """
        
        structured_prompt = PromptTemplate(
            template="""
You are an expert HR analytics assistant. Analyze the following employee survey responses and provide a comprehensive assessment.

Employee Survey Responses:
{survey_responses}

Please provide your analysis in the following EXACT JSON format (no additional text or markdown):

{{
    "positive_sentiment": 70,
    "neutral_sentiment": 20,
    "negative_sentiment": 10,
    "summary_opinion": "Employee shows high satisfaction with team support and benefits",
    "key_positive_1": "Strong team support and feeling valued",
    "key_positive_2": "Comprehensive benefits package",
    "key_positive_3": "Effective onboarding process",
    "attrition_factor_1": "Work-life balance",
    "attrition_problem_1": "Desire for more flexible working hours",
    "retention_strategy_1": "Implement flexible working hour policies",
    "attrition_factor_2": "Career growth",
    "attrition_problem_2": "No specific concerns mentioned but worth monitoring",
    "retention_strategy_2": "Regular career development discussions",
    "attrition_factor_3": "Recognition",
    "attrition_problem_3": "Currently feels valued but maintain recognition programs",
    "retention_strategy_3": "Continue acknowledgment programs and peer recognition"
}}

Guidelines:
- Sentiment percentages must add up to 100
- Be specific and actionable in retention strategies
- Provide insights that HR can immediately act upon
""",
            input_variables=["survey_responses"]
        )
        
        chain = structured_prompt | model
        result = chain.invoke({"survey_responses": sample_responses})
        
        print("\n✅ Sentiment analysis test successful!")
        print("Raw response:")
        print(result.content)
        
        # Try to parse as JSON
        try:
            # Clean up response
            response_text = result.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            parsed = json.loads(response_text)
            print("\n✅ JSON parsing successful!")
            print(f"Sentiment breakdown: Positive: {parsed.get('positive_sentiment')}%, "
                  f"Neutral: {parsed.get('neutral_sentiment')}%, "
                  f"Negative: {parsed.get('negative_sentiment')}%")
            return True
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print("But Ollama response was received - may need prompt tuning")
            return True
            
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Ollama Integration...")
    print(f"Base URL: {OLLAMA_BASE_URL}")
    print(f"Model: {OLLAMA_MODEL}")
    print("-" * 50)
    
    # Test basic connection
    connection_ok = test_ollama_connection()
    
    if connection_ok:
        print("\n" + "-" * 50)
        # Test sentiment analysis
        test_sentiment_analysis()
    
    print("\n🏁 Test completed!")
