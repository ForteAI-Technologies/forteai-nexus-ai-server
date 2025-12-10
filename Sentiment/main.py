import os
try:
    # Prefer dotenv in development, but don't fail if it's not installed in the environment
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import mysql.connector
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import itertools
import logging
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= DB CONNECTION =================
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'forte_hr'),
        port=int(os.getenv('DB_PORT', 3306))
    )

# ================= SURVEY TEMPLATE =================
SURVEY_TEMPLATE = """
# Employee Retention & Sentiment Survey

1. Let's start simple: If you were describing what it's like working here to a friend, what would you say?
Answer: {q1}

2. On a scale from 1 (very unhappy) to 5 (very happy), where would you put yourself?
Answer: {q2}

3. What's something about your compensation or benefits that makes you feel good?
Answer: {q3}

4. If you could wave a magic wand and fix something about how you're rewarded here, what would you change?
Answer: {q4}

5. Looking back, how well did your onboarding set you up for success?
Answer: {q5}

6. Can you tell me about a particular moment you felt supported as a new hire?
Answer: {q6}

7. Have you had real opportunities to build new skills or step up?
Answer: {q7}

8. Who's your go-to for career advice or mentoring around here?
Answer: {q8}

9. How much do you feel your manager is in your corner day-to-day?
Answer: {q9}

10. Think back—when did you last get feedback that actually made a difference for you?
Answer: {q10}

11. How much do you get to decide how you tackle your work?
Answer: {q11}

12. Is there anything that frustrates or holds you back at work?
Answer: {q12}

13. How's your work-life balance holding up these days?
Answer: {q13}

14. Are there company programs or benefits that make your life easier—or ones you wish we had?
Answer: {q14}

15. How comfortable are you being yourself here?
Answer: {q15}

16. What's your team culture like—what do you love or wish was better?
Answer: {q16}

17. How well do people team up and help each other here?
Answer: {q17}

18. How well do you feel kept in the loop and involved?
Answer: {q18}

19. How much do you trust leaders to look out for employees' interests?
Answer: {q19}

20. Share a story when you felt recognized or appreciated here.
Answer: {q20}

21. Are promotions and rewards at this company handled in a way that feels fair to you?
Answer: {q21}

22. Have you ever experienced stress or burnout on the job? What did you do, or what support do you wish you'd had?
Answer: {q22}

23. Do you have a clear sense of next steps and growth for your career here?
Answer: {q23}

24. What motivates you to show up to work?
Answer: {q24}

25. What's your one wish for making this company a better place to work?
Answer: {q25}
"""

# ================= LLM SETUP =================
# --- API Key Rotation ---
# (Kept for backward compatibility but not required for Ollama)
_raw_keys = os.getenv('GOOGLE_API_KEYS')
API_KEYS = [k.strip() for k in _raw_keys.split(',') if k.strip()] if _raw_keys else []
key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None
request_count = 0
REQUEST_LIMIT = int(os.getenv('GOOGLE_API_REQUEST_LIMIT', 15))
CURRENT_API_KEY = API_KEYS[0] if API_KEYS else os.getenv('GOOGLE_API_KEY', '')

# OLLAMA configuration (the user requested these values)
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:latest')

def get_api_key():
    """Return an API key. Rotates through configured keys when REQUEST_LIMIT is exceeded.

    If no keys are configured via environment variables, returns an empty string which
    allows callers to decide how to proceed (the LangChain client will likely raise
    an error if no valid key is provided).
    """
    global request_count, CURRENT_API_KEY, key_cycle
    if not API_KEYS:
        return CURRENT_API_KEY

    request_count += 1
    if request_count > REQUEST_LIMIT:
        try:
            CURRENT_API_KEY = next(key_cycle)
        except Exception:
            # Recreate cycle and continue
            key_cycle = itertools.cycle(API_KEYS)
            CURRENT_API_KEY = next(key_cycle)
        request_count = 1
        logger.info(f"Switching to new API key: ...{CURRENT_API_KEY[-4:]}")
    return CURRENT_API_KEY

# ================= NEW: FLASK WEB SERVICE INTEGRATION =================
app = Flask(__name__)
CORS(app)

# Updated database connection for forteai_nexus database
def get_fortai_db_connection():
    """Updated DB connection for the main ForteAI database"""
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'forteai_nexus'),
        port=int(os.getenv('DB_PORT', 3306))
    )

# Updated structured prompt with stricter JSON formatting requirements
STRUCTURED_ANALYSIS_TEMPLATE = """
CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. You MUST return ONLY a valid JSON object - nothing else
2. NO text before the opening JSON
3. NO text after the closing JSON
4. NO markdown formatting, NO code blocks, NO explanations
5. Return only the JSON object

Analyze this employee's survey and create a JSON response with sentiment analysis.

EMPLOYEE SURVEY:
{survey_responses}

RETURN ONLY THIS JSON STRUCTURE (no other text):
    YOU MUST INCLUDE ALL THESE FIELDS:
positive_sentiment, neutral_sentiment, negative_sentiment, summary_opinion, key_positive_1, key_positive_2, key_positive_3, attrition_factor_1, attrition_problem_1, retention_strategy_1, attrition_factor_2, attrition_problem_2, retention_strategy_2, attrition_factor_3, attrition_problem_3, retention_strategy_3

"""

# ================= COMPANY ANALYSIS TEMPLATE =================
COMPANY_ANALYSIS_TEMPLATE = """
CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. You MUST return ONLY a valid JSON object - nothing else
2. NO text before the opening JSON
3. NO text after the closing JSON  
4. NO markdown formatting, NO code blocks, NO explanations
5. Return only the JSON object
Analyze ALL employee surveys for this company and create a JSON response with company-wide sentiment analysis.

COMPANY EMPLOYEE SURVEY DATA:
{all_employee_data}

RETURN ONLY THIS JSON STRUCTURE (no other text):
    YOU MUST INCLUDE ALL THESE FIELDS:
positive_sentiment, neutral_sentiment, negative_sentiment, summary_opinion, key_positive_1, key_positive_2, key_positive_3, attrition_factor_1, attrition_problem_1, retention_strategy_1, attrition_factor_2, attrition_problem_2, retention_strategy_2, attrition_factor_3, attrition_problem_3, retention_strategy_3



"""

def save_analysis_to_fortai_db(employee_id, company, analysis_data):
    """Save the AI analysis results to responses_langchain_sentiment table"""
    connection = None
    cursor = None

    try:
        connection = get_fortai_db_connection()
        cursor = connection.cursor()

        # Check if record already exists
        check_query = "SELECT id FROM responses_langchain_sentiment WHERE employeesID = %s"
        cursor.execute(check_query, (employee_id,))
        existing_record = cursor.fetchone()

        if existing_record:
            # Update existing record
            update_query = """
            UPDATE responses_langchain_sentiment SET
                company = %s,
                positive_sentiment = %s,
                neutral_sentiment = %s,
                negative_sentiment = %s,
                summary_opinion = %s,
                key_positive_1 = %s,
                key_positive_2 = %s,
                key_positive_3 = %s,
                attrition_factor_1 = %s,
                attrition_problem_1 = %s,
                retention_strategy_1 = %s,
                attrition_factor_2 = %s,
                attrition_problem_2 = %s,
                retention_strategy_2 = %s,
                attrition_factor_3 = %s,
                attrition_problem_3 = %s,
                retention_strategy_3 = %s,
                created_at = CURRENT_TIMESTAMP
            WHERE employeesID = %s
            """

            values = (
                company,
                analysis_data['positive_sentiment'],
                analysis_data['neutral_sentiment'],
                analysis_data['negative_sentiment'],
                analysis_data['summary_opinion'],
                analysis_data['key_positive_1'],
                analysis_data['key_positive_2'],
                analysis_data['key_positive_3'],
                analysis_data['attrition_factor_1'],
                analysis_data['attrition_problem_1'],
                analysis_data['retention_strategy_1'],
                analysis_data['attrition_factor_2'],
                analysis_data['attrition_problem_2'],
                analysis_data['retention_strategy_2'],
                analysis_data['attrition_factor_3'],
                analysis_data['attrition_problem_3'],
                analysis_data['retention_strategy_3'],
                employee_id
            )

            cursor.execute(update_query, values)
            logger.info(f"Updated existing analysis record for employee: {employee_id}")

        else:
            # Insert new record
            insert_query = """
            INSERT INTO responses_langchain_sentiment
            (employeesID, company, positive_sentiment, neutral_sentiment, negative_sentiment,
             summary_opinion, key_positive_1, key_positive_2, key_positive_3,
             attrition_factor_1, attrition_problem_1, retention_strategy_1,
             attrition_factor_2, attrition_problem_2, retention_strategy_2,
             attrition_factor_3, attrition_problem_3, retention_strategy_3)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            values = (
                employee_id,
                company,
                analysis_data['positive_sentiment'],
                analysis_data['neutral_sentiment'],
                analysis_data['negative_sentiment'],
                analysis_data['summary_opinion'],
                analysis_data['key_positive_1'],
                analysis_data['key_positive_2'],
                analysis_data['key_positive_3'],
                analysis_data['attrition_factor_1'],
                analysis_data['attrition_problem_1'],
                analysis_data['retention_strategy_1'],
                analysis_data['attrition_factor_2'],
                analysis_data['attrition_problem_2'],
                analysis_data['retention_strategy_2'],
                analysis_data['attrition_factor_3'],
                analysis_data['attrition_problem_3'],
                analysis_data['retention_strategy_3']
            )

            cursor.execute(insert_query, values)
            logger.info(f"Inserted new analysis record for employee: {employee_id}")

        connection.commit()
        return True

    except mysql.connector.Error as e:
        logger.error(f"Database save error: {e}")
        if connection:
            connection.rollback()
        raise

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def format_survey_responses_for_flask(answers):
    """Format the survey answers for analysis in Flask"""
    formatted_responses = []
    for question_num, answer_data in answers.items():
        # Handle both dict format (new) and string format (legacy)
        if isinstance(answer_data, dict):
            question_text = answer_data.get('question', '')
            answer_text = answer_data.get('answer', '')
            if answer_text and str(answer_text).strip():
                formatted_responses.append(f"{question_text}\nAnswer: {answer_text}\n")
        elif isinstance(answer_data, str) and answer_data.strip():
            formatted_responses.append(f"Question {question_num}: {answer_data}")

    return "\n".join(formatted_responses)

def analyze_sentiment_for_flask(answers):
    """Perform sentiment analysis using Ollama via LangChain with improved error handling"""
    try:
        # Format the survey responses
        survey_text = format_survey_responses_for_flask(answers)

        if not survey_text.strip():
            raise ValueError("No valid survey responses found")

        logger.info("Starting sentiment analysis with structured output using ChatOllama...")

        # Create the ChatOllama model instance
        try:
            model = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"Model initialization error (ChatOllama): {e}")
            raise ValueError(f"Failed to initialize ChatOllama model: {e}")

        # Create structured prompt and LLMChain
        structured_prompt = PromptTemplate(
            template=STRUCTURED_ANALYSIS_TEMPLATE,
            input_variables=["survey_responses"]
        )
        chain = LLMChain(llm=model, prompt=structured_prompt)

        # Generate the analysis with retry logic
        max_attempts = 3
        analysis_data = None
        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_attempts}")

                analysis_text = chain.predict(survey_responses=survey_text).strip()

                logger.info(f"Raw AI response length: {len(analysis_text)}")
                logger.info(f"Raw AI response preview: {analysis_text[:200]}...")

                if not analysis_text:
                    raise ValueError("Empty response from AI model")

                # Clean up any markdown formatting
                if analysis_text.startswith('```json'):
                    analysis_text = analysis_text[7:]
                    if analysis_text.endswith('```'):
                        analysis_text = analysis_text[:-3]
                elif analysis_text.startswith('```'):
                    analysis_text = analysis_text[3:]
                    if analysis_text.endswith('```'):
                        analysis_text = analysis_text[:-3]

                # Remove any extra whitespace or newlines
                analysis_text = analysis_text.strip()

                # More aggressive JSON cleanup
                # Remove any text before the first {
                first_brace = analysis_text.find('{')
                if first_brace > 0:
                    analysis_text = analysis_text[first_brace:]

                # Remove any text after the last }
                last_brace = analysis_text.rfind('}')
                if last_brace > 0 and last_brace < len(analysis_text) - 1:
                    analysis_text = analysis_text[:last_brace + 1]

                # Remove any control characters that break JSON
                analysis_text = ''.join(char for char in analysis_text if ord(char) >= 32 or char in '\n\r\t')

                # Try to parse JSON
                try:
                    analysis_data = json.loads(analysis_text)
                    logger.info("Successfully parsed JSON response")
                    break  # Success, exit retry loop

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error on attempt {attempt + 1}: {e}")
                    logger.error(f"Cleaned response: {analysis_text}")

                    if attempt == max_attempts - 1:  # Last attempt
                        # Try to extract JSON from partial response
                        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                        if json_match:
                            try:
                                analysis_data = json.loads(json_match.group())
                                logger.info("Successfully extracted JSON from partial response")
                                break
                            except:
                                pass
                        raise ValueError(f"Invalid JSON response after {max_attempts} attempts")

                    # Wait before retry
                    import time
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Generation error on attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    raise

        if analysis_data is None:
            raise ValueError("Failed to generate valid JSON analysis from the model")

        # Validate required fields
        required_fields = [
            'positive_sentiment' , 'neutral_sentiment', 'negative_sentiment',
            'summary_opinion', 'key_positive_1', 'key_positive_2', 'key_positive_3',
            'attrition_factor_1', 'attrition_problem_1', 'retention_strategy_1',
            'attrition_factor_2', 'attrition_problem_2', 'retention_strategy_2',
            'attrition_factor_3', 'attrition_problem_3', 'retention_strategy_3'
        ]

        missing_fields = [field for field in required_fields if field not in analysis_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate and normalize sentiment percentages
        try:
            pos_sent = int(analysis_data['positive_sentiment'])
            neu_sent = int(analysis_data['neutral_sentiment'])
            neg_sent = int(analysis_data['negative_sentiment'])

            total_sentiment = pos_sent + neu_sent + neg_sent

            if abs(total_sentiment - 100) > 10:  # Allow 10% tolerance
                logger.warning(f"Sentiment percentages don't add up to 100: {total_sentiment}. Normalizing...")
                # Normalize the percentages
                if total_sentiment > 0:
                    factor = 100 / total_sentiment
                    analysis_data['positive_sentiment'] = int(pos_sent * factor)
                    analysis_data['neutral_sentiment'] = int(neu_sent * factor)
                    analysis_data['negative_sentiment'] = 100 - analysis_data['positive_sentiment'] - analysis_data['neutral_sentiment']
                else:
                    # Default distribution if all zero
                    analysis_data['positive_sentiment'] = 40
                    analysis_data['neutral_sentiment'] = 40
                    analysis_data['negative_sentiment'] = 20

        except (ValueError, TypeError) as e:
            logger.error(f"Sentiment validation error: {e}")
            # Provide default values
            analysis_data['positive_sentiment'] = 40
            analysis_data['neutral_sentiment'] = 40
            analysis_data['negative_sentiment'] = 20

        # Ensure all text fields are strings and not empty
        text_fields = [
            'summary_opinion', 'key_positive_1', 'key_positive_2', 'key_positive_3',
            'attrition_factor_1', 'attrition_problem_1', 'retention_strategy_1',
            'attrition_factor_2', 'attrition_problem_2', 'retention_strategy_2',
            'attrition_factor_3', 'attrition_problem_3', 'retention_strategy_3'
        ]

        for field in text_fields:
            if not isinstance(analysis_data.get(field), str) or not analysis_data[field].strip():
                analysis_data[field] = f"Analysis needed for {field.replace('_', ' ')}"

        logger.info("Sentiment analysis completed successfully")
        return analysis_data

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        # Return a fallback analysis structure
        return {
            "positive_sentiment": 50,
            "neutral_sentiment": 30,
            "negative_sentiment": 20,
            "summary_opinion": f"Analysis could not be completed due to technical issues: {str(e)}",
            "key_positive_1": "Unable to determine - technical issue",
            "key_positive_2": "Unable to determine - technical issue",
            "key_positive_3": "Unable to determine - technical issue",
            "attrition_factor_1": "Technical analysis failure",
            "attrition_problem_1": "System unable to process survey responses",
            "retention_strategy_1": "Manual review required",
            "attrition_factor_2": "Data processing error",
            "attrition_problem_2": "AI analysis service unavailable",
            "retention_strategy_2": "Retry analysis when service is restored",
            "attrition_factor_3": "Service interruption",
            "attrition_problem_3": "Temporary technical difficulties",
            "retention_strategy_3": "Contact IT support for resolution"
        }

def get_company_employee_data(company_id):
    """Fetch ALL employee RAW survey responses for a specific company for comprehensive analysis"""
    connection = None
    cursor = None

    try:
        connection = get_fortai_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Get all employees in the company (excluding HR)
        employees_query = """
        SELECT employeesID, name
        FROM employees
        WHERE company_id = %s AND role != 'HR' AND COALESCE(is_filled, 0) = 1
        ORDER BY employeesID
        """

        cursor.execute(employees_query, (company_id,))
        employees = cursor.fetchall()

        if not employees:
            logger.warning(f"No employees found for company_id: {company_id}")
            return []

        logger.info(f"Found {len(employees)} employees for company {company_id}")

        # For each employee, fetch all their survey responses
        employee_data = []

        for employee in employees:
            emp_id = employee['employeesID']
            emp_name = employee['name']

            # Fetch all responses for this employee with question details
            responses_query = """
            SELECT
                rs.form_question_id,
                rs.answer_text,
                rs.answer_choice,
                fq.master_question_id,
                mq.question_number,
                fq.question_text
            FROM responses_sentiment rs
            JOIN formquestions_sentiment fq ON rs.form_question_id = fq.form_question_id
            JOIN masterquestions_sentiment mq ON fq.master_question_id = mq.master_question_id
            WHERE rs.employeesID = %s
            ORDER BY mq.question_number
            """

            cursor.execute(responses_query, (emp_id,))
            responses = cursor.fetchall()

            if not responses:
                logger.warning(f"No responses found for employee {emp_id}")
                continue

            # Format responses into a structured dictionary
            formatted_responses = {}
            for response in responses:
                q_num = response['question_number']
                question_text = response['question_text']
                answer = response['answer_text'] if response['answer_text'] else response['answer_choice']

                formatted_responses[f"q{q_num}"] = {
                    "question": question_text,
                    "answer": answer or "No response"
                }

            employee_data.append({
                "employeesID": emp_id,
                "name": emp_name,
                "responses": formatted_responses
            })

        logger.info(f"Successfully fetched survey data for {len(employee_data)} employees")
        return employee_data

    except mysql.connector.Error as e:
        logger.error(f"Database error getting company employee data: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def format_company_data_for_analysis(employee_data):
    """Format ALL employee RAW survey responses for comprehensive company analysis"""
    if not employee_data:
        return "No employee survey data available for analysis."

    formatted_data = []
    formatted_data.append("="*80)
    formatted_data.append(f"COMPANY-WIDE SENTIMENT SURVEY DATA - {len(employee_data)} EMPLOYEES")
    formatted_data.append("="*80)

    for idx, emp in enumerate(employee_data, 1):
        emp_section = [
            f"\n{'='*80}",
            f"EMPLOYEE #{idx} - ID: {emp['employeesID']} - Name: {emp.get('name', 'Unknown')}",
            f"{'='*80}",
            ""
        ]

        responses = emp.get('responses', {})

        # Format all 25 questions and answers
        for q_key in sorted(responses.keys(), key=lambda x: int(x[1:])):
            q_data = responses[q_key]
            question_text = q_data.get('question', 'Unknown question')
            answer_text = q_data.get('answer', 'No response')

            emp_section.append(f"{q_key.upper()}. {question_text}")
            emp_section.append(f"ANSWER: {answer_text}")
            emp_section.append("")

        formatted_data.extend(emp_section)

    formatted_data.append("="*80)
    formatted_data.append(f"END OF SURVEY DATA - TOTAL EMPLOYEES ANALYZED: {len(employee_data)}")
    formatted_data.append("="*80)

    return "\n".join(formatted_data)

def analyze_company_sentiment(company_id):
    """Perform company-wide sentiment analysis using ChatOllama via LangChain"""
    try:
        # Get all employee data for the company
        employee_data = get_company_employee_data(company_id)

        if not employee_data:
            raise ValueError(f"No employee sentiment data found for company_id: {company_id}")

        # Format the data for analysis
        formatted_company_data = format_company_data_for_analysis(employee_data)

        logger.info(f"Starting company sentiment analysis for {len(employee_data)} employees...")
        logger.info(f"Sample of data being sent to AI: {formatted_company_data[:500]}...")  # Log first 500 chars

        # Create the ChatOllama model instance
        try:
            model = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"Model initialization error (ChatOllama): {e}")
            raise ValueError(f"Failed to initialize ChatOllama model: {e}")

        # Create structured prompt with explicit JSON instruction
        company_prompt = PromptTemplate(
            template="You must return valid JSON only. " + COMPANY_ANALYSIS_TEMPLATE,
            input_variables=["all_employee_data"]
        )

        chain = LLMChain(llm=model, prompt=company_prompt)

        # Generate the analysis with retry logic
        max_attempts = 3
        analysis_data = None
        for attempt in range(max_attempts):
            try:
                logger.info(f"Company analysis attempt {attempt + 1} of {max_attempts}")

                analysis_text = chain.predict(all_employee_data=formatted_company_data).strip()

                # Get the response content
                logger.info(f"Raw company AI response length: {len(analysis_text)}")
                logger.info(f"Raw company AI response first 200 chars: {analysis_text[:200]}")

                if not analysis_text:
                    raise ValueError("Empty response from AI model")

                # Clean up any markdown formatting and extra text
                if analysis_text.startswith('```json'):
                    analysis_text = analysis_text[7:]
                    if analysis_text.endswith('```'):
                        analysis_text = analysis_text[:-3]
                elif analysis_text.startswith('```'):
                    analysis_text = analysis_text[3:]
                    if analysis_text.endswith('```'):
                        analysis_text = analysis_text[:-3]

                # Remove any leading/trailing whitespace and newlines
                analysis_text = analysis_text.strip()

                # More aggressive JSON cleanup
                # Remove any text before the first {
                first_brace = analysis_text.find('{')
                if first_brace > 0:
                    analysis_text = analysis_text[first_brace:]

                # Remove any text after the last }
                last_brace = analysis_text.rfind('}')
                if last_brace > 0 and last_brace < len(analysis_text) - 1:
                    analysis_text = analysis_text[:last_brace + 1]

                # Remove any control characters that break JSON
                analysis_text = ''.join(char for char in analysis_text if ord(char) >= 32 or char in '\n\r\t')

                logger.info(f"Cleaned company response: {analysis_text[:200]}...")

                # Try to parse JSON
                try:
                    analysis_data = json.loads(analysis_text)
                    logger.info("Successfully parsed company JSON response")
                    break

                except json.JSONDecodeError as e:
                    logger.error(f"Company JSON parsing error on attempt {attempt + 1}: {e}")
                    logger.error(f"Problematic JSON text: {analysis_text}")

                    if attempt == max_attempts - 1:
                        # Last attempt - try to fix common JSON issues
                        try:
                            # Fix common issues like trailing commas, unescaped quotes
                            fixed_text = analysis_text
                            # Remove trailing commas before closing braces
                            fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
                            # Try to fix unescaped quotes in strings
                            fixed_text = re.sub(r'(?<!\\)"(?=.*":)', r'\\"', fixed_text)

                            analysis_data = json.loads(fixed_text)
                            logger.info("Successfully parsed company JSON after fixing")
                            break
                        except:
                            logger.error("Failed to parse JSON even after attempted fixes")
                            raise ValueError(f"Invalid JSON response after {max_attempts} attempts: {str(e)}")

                    import time
                    time.sleep(2)  # Longer delay between retries

            except Exception as e:
                logger.error(f"Company generation error on attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    raise

        if analysis_data is None:
            raise ValueError("Failed to generate valid JSON analysis from the model")

        # Validate required fields
        required_fields = [
            'positive_sentiment', 'neutral_sentiment', 'negative_sentiment',
            'summary_opinion', 'key_positive_1', 'key_positive_2', 'key_positive_3',
            'attrition_factor_1', 'attrition_problem_1', 'retention_strategy_1',
            'attrition_factor_2', 'attrition_problem_2', 'retention_strategy_2',
            'attrition_factor_3', 'attrition_problem_3', 'retention_strategy_3'
        ]

        missing_fields = [field for field in required_fields if field not in analysis_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate and normalize sentiment percentages
        try:
            pos_sent = int(analysis_data['positive_sentiment'])
            neu_sent = int(analysis_data['neutral_sentiment'])
            neg_sent = int(analysis_data['negative_sentiment'])

            total_sentiment = pos_sent + neu_sent + neg_sent

            if abs(total_sentiment - 100) > 10:
                logger.warning(f"Company sentiment percentages don't add up to 100: {total_sentiment}. Normalizing...")
                if total_sentiment > 0:
                    factor = 100 / total_sentiment
                    analysis_data['positive_sentiment'] = int(pos_sent * factor)
                    analysis_data['neutral_sentiment'] = int(neu_sent * factor)
                    analysis_data['negative_sentiment'] = 100 - analysis_data['positive_sentiment'] - analysis_data['neutral_sentiment']
                else:
                    analysis_data['positive_sentiment'] = 50
                    analysis_data['neutral_sentiment'] = 30
                    analysis_data['negative_sentiment'] = 20

        except (ValueError, TypeError) as e:
            logger.error(f"Company sentiment validation error: {e}")
            analysis_data['positive_sentiment'] = 50
            analysis_data['neutral_sentiment'] = 30
            analysis_data['negative_sentiment'] = 20

        # Ensure all text fields are strings and not empty
        text_fields = [
            'summary_opinion', 'key_positive_1', 'key_positive_2', 'key_positive_3',
            'attrition_factor_1', 'attrition_problem_1', 'retention_strategy_1',
            'attrition_factor_2', 'attrition_problem_2', 'retention_strategy_2',
            'attrition_factor_3', 'attrition_problem_3', 'retention_strategy_3'
        ]

        for field in text_fields:
            if not isinstance(analysis_data.get(field), str) or not analysis_data[field].strip():
                analysis_data[field] = f"Company analysis needed for {field.replace('_', ' ')}"

        logger.info("Company sentiment analysis completed successfully")
        return analysis_data

    except Exception as e:
        logger.error(f"Company sentiment analysis error: {e}")
        # Return a fallback analysis structure
        return {
            "positive_sentiment": 50,
            "neutral_sentiment": 30,
            "negative_sentiment": 20,
            "summary_opinion": f"Company analysis could not be completed due to technical issues: {str(e)}",
            "key_positive_1": "Unable to determine - technical issue",
            "key_positive_2": "Unable to determine - technical issue",
            "key_positive_3": "Unable to determine - technical issue",
            "attrition_factor_1": "Technical analysis failure",
            "attrition_problem_1": "System unable to process company data",
            "retention_strategy_1": "Manual review required",
            "attrition_factor_2": "Data processing error",
            "attrition_problem_2": "AI analysis service unavailable",
            "retention_strategy_2": "Retry analysis when service is restored",
            "attrition_factor_3": "Service interruption",
            "attrition_problem_3": "Temporary technical difficulties",
            "retention_strategy_3": "Contact IT support for resolution"
        }

def save_company_analysis_to_db(company_id, analysis_data):
    """Save the company AI analysis results to company_reports_sentiment table"""
    connection = None
    cursor = None

    try:
        connection = get_fortai_db_connection()
        cursor = connection.cursor()

        # Check if record already exists
        check_query = "SELECT id FROM company_reports_sentiment WHERE company_id = %s"
        cursor.execute(check_query, (company_id,))
        existing_record = cursor.fetchone()

        if existing_record:
            # Update existing record
            update_query = """
            UPDATE company_reports_sentiment SET
                positive_sentiment = %s,
                neutral_sentiment = %s,
                negative_sentiment = %s,
                summary_opinion = %s,
                key_positive_1 = %s,
                key_positive_2 = %s,
                key_positive_3 = %s,
                attrition_factor_1 = %s,
                attrition_problem_1 = %s,
                retention_strategy_1 = %s,
                attrition_factor_2 = %s,
                attrition_problem_2 = %s,
                retention_strategy_2 = %s,
                attrition_factor_3 = %s,
                attrition_problem_3 = %s,
                retention_strategy_3 = %s,
                created_at = CURRENT_TIMESTAMP,
                is_filled = 1
            WHERE company_id = %s
            """

            values = (
                analysis_data['positive_sentiment'],
                analysis_data['neutral_sentiment'],
                analysis_data['negative_sentiment'],
                analysis_data['summary_opinion'],
                analysis_data['key_positive_1'],
                analysis_data['key_positive_2'],
                analysis_data['key_positive_3'],
                analysis_data['attrition_factor_1'],
                analysis_data['attrition_problem_1'],
                analysis_data['retention_strategy_1'],
                analysis_data['attrition_factor_2'],
                analysis_data['attrition_problem_2'],
                analysis_data['retention_strategy_2'],
                analysis_data['attrition_factor_3'],
                analysis_data['attrition_problem_3'],
                analysis_data['retention_strategy_3'],
                company_id
            )

            cursor.execute(update_query, values)
            logger.info(f"Updated existing company analysis record for company_id: {company_id}")

        else:
            # Insert new record
            insert_query = """
            INSERT INTO company_reports_sentiment
            (company_id, positive_sentiment, neutral_sentiment, negative_sentiment,
             summary_opinion, key_positive_1, key_positive_2, key_positive_3,
             attrition_factor_1, attrition_problem_1, retention_strategy_1,
             attrition_factor_2, attrition_problem_2, retention_strategy_2,
             attrition_factor_3, attrition_problem_3, retention_strategy_3, is_filled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            values = (
                company_id,
                analysis_data['positive_sentiment'],
                analysis_data['neutral_sentiment'],
                analysis_data['negative_sentiment'],
                analysis_data['summary_opinion'],
                analysis_data['key_positive_1'],
                analysis_data['key_positive_2'],
                analysis_data['key_positive_3'],
                analysis_data['attrition_factor_1'],
                analysis_data['attrition_problem_1'],
                analysis_data['retention_strategy_1'],
                analysis_data['attrition_factor_2'],
                analysis_data['attrition_problem_2'],
                analysis_data['retention_strategy_2'],
                analysis_data['attrition_factor_3'],
                analysis_data['attrition_problem_3'],
                analysis_data['retention_strategy_3'],
                1
            )

            cursor.execute(insert_query, values)
            logger.info(f"Inserted new company analysis record for company_id: {company_id}")

        connection.commit()
        return True

    except mysql.connector.Error as e:
        logger.error(f"Database save error for company analysis: {e}")
        if connection:
            connection.rollback()
        raise

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# ================= FLASK ROUTES =================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        connection = get_fortai_db_connection()
        if connection:
            connection.close()
            db_status = "connected"
        else:
            db_status = "disconnected"
    except:
        db_status = "error"

    return jsonify({
        'status': 'healthy',
        'service': 'ForteAI Flask Sentiment Analysis',
        'database': db_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_employee_sentiment_flask():
    """Main endpoint for sentiment analysis - integrates with ForteAI database"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        employee_id = data.get('employeeId')
        company = data.get('company')
        answers = data.get('answers')

        if not employee_id:
            return jsonify({'error': 'employeeId is required'}), 400

        if not company:
            return jsonify({'error': 'company is required'}), 400

        if not answers or not isinstance(answers, dict):
            return jsonify({'error': 'answers dictionary is required'}), 400

        logger.info(f"Starting analysis for employee: {employee_id} from company: {company}")

        # Perform sentiment analysis using ChatOllama via LangChain
        analysis_result = analyze_sentiment_for_flask(answers)

        # Save to ForteAI database
        save_analysis_to_fortai_db(employee_id, company, analysis_result)

        logger.info(f"Analysis completed and saved for employee: {employee_id}")

        return jsonify({
            'success': True,
            'message': 'Sentiment analysis completed and saved successfully',
            'employeeId': employee_id,
            'company': company,
            'analysis': analysis_result,
            'timestamp': datetime.now().isoformat()
        })

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Internal server error occurred during analysis'}), 500

@app.route('/debug/database', methods=['GET'])
def debug_database():
    """Debug endpoint to check database connection and table structure"""
    try:
        connection = get_fortai_db_connection()
        cursor = connection.cursor()

        # Check table structure
        cursor.execute("DESCRIBE responses_langchain_sentiment")
        table_structure = cursor.fetchall()

        # Check sample data
        cursor.execute("SELECT COUNT(*) FROM responses_langchain_sentiment")
        record_count = cursor.fetchone()[0]

        cursor.close()
        connection.close()

        return jsonify({
            'success': True,
            'table_structure': [
                {
                    'field': row[0],
                    'type': row[1],
                    'null': row[2],
                    'key': row[3],
                    'default': row[4],
                    'extra': row[5]
                } for row in table_structure
            ],
            'record_count': record_count,
            'database_config': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'forteai_nexus'),
                'user': os.getenv('DB_USER', 'root')
            }
        })

    except Exception as e:
        logger.error(f"Database debug error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-ai', methods=['POST'])
def test_ai_connection():
    """Test endpoint to verify AI connection and response format"""
    try:
        data = request.get_json()
        test_text = data.get('text', 'This is a test employee survey response about work satisfaction.')

        logger.info(f"Testing ChatOllama at {OLLAMA_BASE_URL} using model {OLLAMA_MODEL}")

        # Test the model directly
        model = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0,
        )

        simple_prompt = "Respond with exactly this JSON: {\"test\": \"success\", \"message\": \"AI connection working\"}"
        prompt = PromptTemplate(template="{input}", input_variables=["input"])
        chain = LLMChain(llm=model, prompt=prompt)
        result = chain.predict(input=simple_prompt)

        response_text = result

        return jsonify({
            'success': True,
            'raw_response': response_text,
            'ollama_base_url': OLLAMA_BASE_URL,
            'ollama_model': OLLAMA_MODEL,
            'test_input': test_text
        })

    except Exception as e:
        logger.error(f"AI test error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-company', methods=['POST'])
def analyze_company_sentiment_flask():
    """Endpoint for company-wide sentiment analysis - integrates with ForteAI database"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        company_id = data.get('companyId') or data.get('company_id')

        if not company_id:
            return jsonify({'error': 'companyId or company_id is required'}), 400

        logger.info(f"Starting company analysis for company_id: {company_id}")

        # Perform company-wide sentiment analysis
        analysis_result = analyze_company_sentiment(company_id)

        # Save to ForteAI database
        save_company_analysis_to_db(company_id, analysis_result)

        logger.info(f"Company analysis completed and saved for company_id: {company_id}")

        return jsonify({
            'success': True,
            'message': 'Company sentiment analysis completed and saved successfully',
            'companyId': company_id,
            'analysis': analysis_result,
            'timestamp': datetime.now().isoformat()
        })

    except ValueError as e:
        logger.error(f"Company validation error: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Company analysis error: {e}")
        return jsonify({'error': 'Internal server error occurred during company analysis'}), 500

def fetch_employee_survey_responses(employee_id, company_name):
    """Fetch existing survey responses for a specific employee"""
    try:
        connection = get_fortai_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT employee_name, work_life_balance, compensation, growth_opportunities,
               management_quality, team_culture, job_satisfaction, company, feedback,
               submission_time, work_life_balance_rating, compensation_rating,
               growth_opportunities_rating, management_quality_rating,
               team_culture_rating, job_satisfaction_rating
        FROM Responses_sentiment
        WHERE employee_name = %s AND company = %s
        ORDER BY submission_time DESC
        LIMIT 1
        """

        cursor.execute(query, (employee_id, company_name))
        result = cursor.fetchone()

        cursor.close()
        connection.close()

        return result

    except Exception as e:
        logger.error(f"Error fetching employee survey responses: {e}")
        return None

@app.route('/regenerate-report', methods=['POST'])
def regenerate_employee_report():
    """Regenerate sentiment analysis report for an existing employee"""
    try:
        data = request.json
        employee_id = data.get('employeeId')
        company_name = data.get('company')

        if not employee_id or not company_name:
            return jsonify({'error': 'Employee ID and company name are required'}), 400

        # Fetch existing survey responses
        employee_data = fetch_employee_survey_responses(employee_id, company_name)

        if not employee_data:
            return jsonify({'error': f'No survey data found for employee {employee_id} in company {company_name}'}), 404

        # Convert to format expected by analysis function
        survey_data = {
            'work_life_balance': employee_data['work_life_balance'],
            'compensation': employee_data['compensation'],
            'growth_opportunities': employee_data['growth_opportunities'],
            'management_quality': employee_data['management_quality'],
            'team_culture': employee_data['team_culture'],
            'job_satisfaction': employee_data['job_satisfaction'],
            'feedback': employee_data['feedback'],
            'work_life_balance_rating': employee_data['work_life_balance_rating'],
            'compensation_rating': employee_data['compensation_rating'],
            'growth_opportunities_rating': employee_data['growth_opportunities_rating'],
            'management_quality_rating': employee_data['management_quality_rating'],
            'team_culture_rating': employee_data['team_culture_rating'],
            'job_satisfaction_rating': employee_data['job_satisfaction_rating']
        }

        # Re-run sentiment analysis
        analysis_result = analyze_sentiment_for_flask(survey_data)

        if not analysis_result:
            return jsonify({'error': 'Failed to regenerate sentiment analysis'}), 500

        # Save the updated analysis to database
        save_success = save_analysis_to_fortai_db(employee_id, company_name, analysis_result)

        if not save_success:
            return jsonify({'error': 'Failed to save regenerated analysis to database'}), 500

        logger.info(f"Successfully regenerated report for employee {employee_id} in company {company_name}")

        return jsonify({
            'message': f'Successfully regenerated report for {employee_id}',
            'employee_name': employee_id,
            'company': company_name,
            'analysis': analysis_result,
            'regenerated_at': datetime.now().isoformat()
        }), 200

    except ValueError as e:
        logger.error(f"Regenerate report validation error: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Regenerate report error: {e}")
        return jsonify({'error': 'Internal server error occurred during report regeneration'}), 500

# ================= FLASK APPLICATION STARTUP =================
if __name__ == "__main__":
    print("🚀 Starting ForteAI Flask Sentiment Analysis Service...")
    print(f"📊 Database: {os.getenv('DB_HOST', 'localhost')}/{os.getenv('DB_NAME', 'forteai_nexus')}")
    print(f"🤖 Ollama Base URL: {OLLAMA_BASE_URL}")
    print(f"🧠 Ollama Model: {OLLAMA_MODEL}")
    print(f"🌐 Server will run on: http://localhost:{int(os.getenv('FLASK_PORT', 5000))}")
    print('GOOGLE_API_KEYS:', os.getenv('GOOGLE_API_KEYS'))
    print('GOOGLE_API_KEY:', os.getenv('GOOGLE_API_KEY'))

    app.run(
        host='0.0.0.0',
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )
