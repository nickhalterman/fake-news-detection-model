from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()


def fact_check(headline):
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Is this news headline factually plausible? Reply on one line in exactly this format: TRUE or FAKE, a period, then one short reason. No markdown.\n\nHeadline: {headline}"
            }
        ],
    )
    answer = "".join(block.text for block in response.content if block.type == "text").strip()
    is_true = answer.replace("*", "").upper().startswith("TRUE")
    return is_true, answer
