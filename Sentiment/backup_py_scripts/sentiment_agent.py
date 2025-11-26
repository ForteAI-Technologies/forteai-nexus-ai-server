import os
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

# Read API key from environment. Set GOOGLE_API_KEY in your environment.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_tokens=800,
)

prompt = PromptTemplate(
    input_variables=["factors", "strategies", "response", "style_context"],
    template="""
                You are an exper HR analytics assistant with deep experience in employee retention and sentiment analysis.
    Analyze the following employee surey responses(more than one) below and provide detailled, actionable insights. Be specific and also focus on practical recommendations.
    Style Guide and Exemplars (adhere strictly to this tone and structure; if conflicts arise, the explicit template below wins):
    {style_context}
    CONTEXT:
    Attrition factors to consider:{factors}
    Available Retention strategies: {strategies}
    Employee Survey Response:
    {response}
    Provide your analysis in this EXACT format (start with "Of course. As an HR analytics assistant..."):
    Of course. As an HR analytics assistant, here is an analysis of the provided employee survey response.
    ### **1. Sentiment Analysis**

Based on a detailed analysis of the survey responses, the employee's sentiment is broken down as follows:

* **Positive: [X]%**
* **Negative: [X]%**
* **Neutral: [X]%**
### **2. Summary of Employee Opinion**

[Write a comprehensive 2-3 sentence assessment of this employee - their engagement level, value to the organization, and overall risk level]

**Key Positives (What's Working):**

* **[Specific Area 1]:** [Detailed explanation of what the employee values and why this is a retention driver]
* **[Specific Area 2]:** [Detailed explanation of what the employee values and why this is a retention driver]
* **[Specific Area 3]:** [Detailed explanation of what the employee values and why this is a retention driver]

**Key Areas for Improvement / Attrition Risks:**

[Write an introductory paragraph explaining how the employee's concerns align with known attrition factors and the opportunities for targeted retention strategies]
-make sure you write no more than 2 sentences under each attrition factor
1. **Attrition Factor: [Primary Concern from the factors list]**
   * **Problem:** [Detailed, specific description of the issue based on the survey response]
   * **Suggested Retention Strategy:** [Specific strategy from the available list with implementation details]

2. **Attrition Factor: [Secondary Concern from the factors list]**
   * **Problem:** [Detailed, specific description of the issue based on the survey response]
   * **Suggested Retention Strategy:** [Specific strategy from the available list with implementation details]

3. **Attrition Factor: [Third Concern from the factors list if applicable]**
   * **Problem:** [Detailed, specific description of the issue based on the survey response]
   * **Suggested Retention Strategy:** [Specific strategy from the available list with implementation details]

IMPORTANT GUIDELINES:
- Ensure percentages add up to 100%
- Be specific and reference actual content from the survey
- Focus on actionable insights
- Keep response between 450-500 words **
- Use professional HR language
- Make clear connections between problems and suggested strategies
"""
)
chain = prompt | model

def analyze_employee_sentiment(factors,strategies,response):
    try:
        # Ensure the style guide is present in the vector store and retrieve relevant style context
        upsert_style_guide()
        style_context = get_style_context(
            query="HR analytics assistant sentiment report style with fixed headings and tone",
            k=3,
        )

        result = chain.invoke(
            {
                "factors": factors,
                "strategies": strategies,
                "response": response,
                "style_context": style_context,
            }
        )
        # Optionally learn from the generated output to reinforce consistency over time
        save_output_example(result.content)
        return result.content
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"
if __name__ == "__main__":
    retention_strategies = "Offer fair pay, Recruit competitively, Hire smarter, Improve onboarding, Provide benefits people want, Invest in professional development, Create pathways for growth, Offer mentorship programs, Train managers to retain, Build employee engagement, Communicate transparently, Offer incentives, Value DE&I, Provide continuous feedback, Work on culture continuously, Engage in CSR programs, Provide autonomy and choice, Consider work-life balance, Emphasize teamwork, Create employee stock ownership plans, Invest in change management, Support employee well-being, Acknowledge hard work and milestones, Be aware of burnout, Know when to let employees go, Flexible hours, Recognition programs, Wellness benefits, Team building, Leadership development"
    attrition_factors = "Work-life balance, Compensation, Management quality, Limited career growth, Lack of recognition, Poor team culture, Low leadership trust, Burnout, Few development opportunities, Perceived unfairness, Job insecurity, Lack of flexibility, Inadequate benefits, Poor communication, Mismatch of role and skills"
    print("=== HR Sentiment Analysis Agent ===")
    print("Enter employee survey response (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    survey_response = "\n".join(lines)

    if survey_response.strip():
        print("\n🔄 Analyzing response...")
        output = analyze_employee_sentiment(attrition_factors, retention_strategies, survey_response)
        print("\n--- Analysis Result ---\n")
        print(output)
    else:
        print("No response provided. Exiting...")
