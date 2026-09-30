from src.fact_verification import verify_news


news = """
NASA announced that humans landed on Mars in 2025.
The astronauts successfully returned to Earth after completing
a three-month mission on the Martian surface.
"""


result = verify_news(news)

print("\n")
print("=" * 80)
print("FACT VERIFICATION RESULT")
print("=" * 80)
print(result)
print("=" * 80)