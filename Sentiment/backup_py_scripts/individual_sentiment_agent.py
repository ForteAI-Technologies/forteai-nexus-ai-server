import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

try:
    from ..style_memory import (
        get_style_context,
        upsert_style_guide,
        save_output_example,
    )
except Exception:
    from style_memory import (
        get_style_context,
        upsert_style_guide,
        save_output_example,
    )

_key = os.getenv('GOOGLE_API_KEY', '')

def _create_model_instance():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv('GOOGLE_API_KEY', _key),
        temperature=0,
        max_tokens=1500,
    )

# Template for individual employee
individual_prompt = PromptTemplate(
    input_variables=["factors", "strategies", "response", "style_context"],
    template="""
You are an expert HR analytics assistant.
Analyze the following employee survey response and produce a detailed report (500–600 words) in the exact structure below.

Style Guide and Exemplars:
{style_context}

Attrition factors to consider: {factors}
Available Retention strategies: {strategies}

### **Individual Employee Report**

Of course. As an HR analytics assistant, here is an analysis of the provided employee survey response.

### **1. Sentiment Analysis**
* **Positive: [X]%**
* **Negative: [X]%**
* **Neutral: [X]%**
--make sure to only just give the percentages without any additional commentary
### **2. Summary of Employee Opinion**
[2–3 sentences]

**Key Positives (What's Working):**
* **[Area 1]:** [Explanation]
* **[Area 2]:** [Explanation]
* **[Area 3]:** [Explanation]

**Key Areas for Improvement / Attrition Risks:**

1. **Attrition Factor: [Primary Concern]**
   * **Problem:** [Specific issue]
   * **Suggested Retention Strategy:** [Actionable solution]

2. **Attrition Factor: [Secondary Concern]**
   * **Problem:** [Specific issue]
   * **Suggested Retention Strategy:** [Actionable solution]

3. **Attrition Factor: [Optional Third Concern]**
   * **Problem:** [Specific issue]
   * **Suggested Retention Strategy:** [Actionable solution]

---
Employee Survey Response:
{response}
"""
)

def analyze_single_employee(factors, strategies, response_text):
    try:
        upsert_style_guide()
        style_context = get_style_context(
            query="HR analytics assistant sentiment report style with fixed headings and tone",
            k=3,
        )

        model = _create_model_instance()
        chain = individual_prompt | model
        result = chain.invoke({
            "factors": factors,
            "strategies": strategies,
            "response": response_text,
            "style_context": style_context,
        })

        save_output_example(result.content)
        return result.content

    except Exception as e:
        return f"Error analyzing individual employee: {str(e)}"


if __name__ == "__main__":
    retention_strategies = "Offer fair pay, Recruit competitively, Hire smarter, Improve onboarding, Provide benefits people want, Invest in professional development, Create pathways for growth, Offer mentorship programs, Train managers to retain, Build employee engagement, Communicate transparently, Offer incentives, Value DE&I, Provide continuous feedback, Work on culture continuously, Engage in CSR programs, Provide autonomy and choice, Consider work-life balance, Emphasize teamwork, Create employee stock ownership plans, Invest in change management, Support employee well-being, Acknowledge hard work and milestones, Be aware of burnout, Know when to let employees go, Flexible hours, Recognition programs, Wellness benefits, Team building, Leadership development"
    attrition_factors = "Work-life balance, Compensation, Management quality, Limited career growth, Lack of recognition, Poor team culture, Low leadership trust, Burnout, Few development opportunities, Perceived unfairness, Job insecurity, Lack of flexibility, Inadequate benefits, Poor communication, Mismatch of role and skills"

    print("=== Individual HR Sentiment Analysis ===")
    print("Enter single employee survey response (press Enter twice to finish):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    survey_response = "\n".join(lines)

    if survey_response.strip():
        print("\n🔄 Analyzing single employee response...\n")
        output = analyze_single_employee(attrition_factors, retention_strategies, survey_response)
        print("\n--- Individual Employee Report ---\n")
        print(output)
    else:
        print("No response provided. Exiting...")
