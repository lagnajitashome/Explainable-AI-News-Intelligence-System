import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please check your .env file."
    )

# Initialize Groq client
client = Groq(api_key=api_key)


def verify_news(news_text):
    """
    Verify factual claims in a news article using
    Groq LLM + web search.
    """

    prompt = f"""
You are an evidence-based news fact verification assistant.

Analyze the following news article.

Your tasks are:

1. Identify the major factual claims in the article.
2. Search the web for reliable evidence for those claims.
3. Compare each claim with the retrieved evidence.
4. Do not assume a claim is true merely because it appears in the article.
5. Do not label a claim false merely because evidence could not be found.
6. Clearly distinguish evidence from uncertainty.

For each major claim, provide:

CLAIM:
The factual statement being checked.

VERDICT:
Choose exactly one:
- SUPPORTED
- PARTIALLY SUPPORTED
- CONTRADICTED
- INSUFFICIENT EVIDENCE

REASON:
Briefly explain why the evidence supports,
partially supports, contradicts, or does not
sufficiently establish the claim.

SOURCES:
List the important sources used to evaluate the claim.

After evaluating the claims, provide:

OVERALL ASSESSMENT:
Give a concise summary of the evidence across
the major claims.

IMPORTANT:
This is evidence-based verification.
Do not claim that the system provides absolute truth.
If reliable evidence is conflicting or insufficient,
explicitly state that.

NEWS ARTICLE:
{news_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        tools=[
            {
                "type": "browser_search"
            }
        ],
        tool_choice="required",
        reasoning_effort="low"
    )

    return response.choices[0].message.content