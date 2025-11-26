import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

try:
    from ..style_memory import (
        get_style_context,
        upsert_style_guide,
    )
except Exception:
    from style_memory import (
        get_style_context,
        upsert_style_guide,
    )

import os
# Read API key from environment. Set GOOGLE_API_KEY in your environment.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_tokens=1500,
)

# Template for combined organizational report
combined_prompt = PromptTemplate(
    input_variables=["factors", "strategies", "all_responses", "style_context"],
    template="""
You are an expert HR analytics assistant.
You are given multiple employee reports. Write ONE combined organizational report (500–600 words) following the structure below.

Style Guide and Exemplars:
{style_context}

Attrition factors to consider: {factors}
Available Retention strategies: {strategies}

### **Combined Organizational Report**

- Aggregate sentiment breakdown (averages across all responses).
- Company-wide summary of employee morale & engagement.
- 3–5 key positives that are consistent.
- 3–5 systemic risks aligned with attrition factors.
- Actionable company-wide recommendations.

Employee Reports to Synthesize:
{all_responses}
"""
)

def generate_combined_report(factors, strategies, individual_reports_text):
    try:
        upsert_style_guide()
        style_context = get_style_context(
            query="HR analytics assistant sentiment report style with fixed headings and tone",
            k=3,
        )

        chain = combined_prompt | model
        combined_result = chain.invoke({
            "factors": factors,
            "strategies": strategies,
            "all_responses": individual_reports_text,
            "style_context": style_context,
        })

        return combined_result.content

    except Exception as e:
        return f"Error generating combined report: {str(e)}"


if __name__ == "__main__":
    retention_strategies = "Offer fair pay, Recruit competitively, Hire smarter, Improve onboarding, Provide benefits people want, Invest in professional development, Create pathways for growth, Offer mentorship programs, Train managers to retain, Build employee engagement, Communicate transparently, Offer incentives, Value DE&I, Provide continuous feedback, Work on culture continuously, Engage in CSR programs, Provide autonomy and choice, Consider work-life balance, Emphasize teamwork, Create employee stock ownership plans, Invest in change management, Support employee well-being, Acknowledge hard work and milestones, Be aware of burnout, Know when to let employees go, Flexible hours, Recognition programs, Wellness benefits, Team building, Leadership development"
    attrition_factors = "Work-life balance, Compensation, Management quality, Limited career growth, Lack of recognition, Poor team culture, Low leadership trust, Burnout, Few development opportunities, Perceived unfairness, Job insecurity, Lack of flexibility, Inadequate benefits, Poor communication, Mismatch of role and skills"

    print("=== Combined Organizational HR Report Generator ===")
    print("Paste all **individual employee reports** (press Enter twice to finish):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    individual_reports_text = "\n".join(lines)

    if individual_reports_text.strip():
        print("\n🔄 Generating organizational report...\n")
        output = generate_combined_report(attrition_factors, retention_strategies, individual_reports_text)
        print("\n--- Combined Organizational Report ---\n")
        print(output)
    else:
        print("No individual reports provided. Exiting...")
