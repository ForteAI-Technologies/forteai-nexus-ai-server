import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

try:
    # Package import (when used as module)
    from ..style_memory import (
        get_style_context,
        upsert_style_guide,
        save_output_example,
    )
except Exception:
    # Script import (when running this file directly)
    from style_memory import (
        get_style_context,
        upsert_style_guide,
        save_output_example,
    )

_key = os.getenv('GOOGLE_API_KEY', '')

def _create_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv('GOOGLE_API_KEY', _key),
        temperature=0,
        max_tokens=3000,  # allow more room for multiple employees
    )

# New prompt for multiple employees in one go
multi_employee_prompt = PromptTemplate(
    input_variables=["factors", "strategies", "responses", "style_context"],
    template="""
You are an expert HR analytics assistant.
You are given multiple employee survey responses.

For EACH employee, produce a **separate analysis report** strictly in the format below.
Do NOT merge employees into one report — treat each response independently.

Style Guide and Exemplars:
{style_context}

Attrition factors to consider: {factors}
Available Retention strategies: {strategies}

### **Individual Employee Report (for each employee)**

Of course. As an HR analytics assistant, here is an analysis of the provided employee survey response.

### **1. Sentiment Analysis**
* **Positive: [X]%**
* **Negative: [X]%**
* **Neutral: [X]%**

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

Employee Survey Responses (analyze each one separately):
{responses}
"""
)


def analyze_employee_sentiment(factors, strategies, responses_text):
    try:
        upsert_style_guide()
        style_context = get_style_context(
            query="HR analytics assistant sentiment report style with fixed headings and tone",
            k=3,
        )

        model = _create_model()
        chain = multi_employee_prompt | model
        result = chain.invoke({
            "factors": factors,
            "strategies": strategies,
            "responses": responses_text,  # full block of text (not split)
            "style_context": style_context,
        })

        # Save for style memory training
        save_output_example(result.content)

        return result.content

    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"


if __name__ == "__main__":
    retention_strategies = (
        "Offer fair pay, Recruit competitively, Hire smarter, Improve onboarding, "
        "Provide benefits people want, Invest in professional development, "
        "Create pathways for growth, Offer mentorship programs, Train managers to retain, "
        "Build employee engagement, Communicate transparently, Offer incentives, "
        "Value DE&I, Provide continuous feedback, Work on culture continuously, "
        "Engage in CSR programs, Provide autonomy and choice, Consider work-life balance, "
        "Emphasize teamwork, Create employee stock ownership plans, Invest in change management, "
        "Support employee well-being, Acknowledge hard work and milestones, Be aware of burnout, "
        "Know when to let employees go, Flexible hours, Recognition programs, "
        "Wellness benefits, Team building, Leadership development"
    )

    attrition_factors = (
        "Work-life balance, Compensation, Management quality, Limited career growth, "
        "Lack of recognition, Poor team culture, Low leadership trust, Burnout, "
        "Few development opportunities, Perceived unfairness, Job insecurity, "
        "Lack of flexibility, Inadequate benefits, Poor communication, "
        "Mismatch of role and skills"
    )

    print("=== HR Sentiment Analysis Agent ===")
    print("Enter employee survey responses (paste all, press Enter twice to finish):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    survey_response = "\n".join(lines)

    if survey_response.strip():
        print("\n🔄 Analyzing responses for ALL employees...\n")
        output = analyze_employee_sentiment(attrition_factors, retention_strategies, survey_response)
        print("\n--- Analysis Result ---\n")
        print(output)
    else:
        print("No response provided. Exiting...")
