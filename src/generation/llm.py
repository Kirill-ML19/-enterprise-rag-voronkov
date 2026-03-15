import os
import asyncio
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLM:
    def __init__(
        self,
        model: str = "z-ai/glm-4.6",
        max_tokens: int = 512,
        temperature: float = 0.0
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in .env")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = getattr(getattr(response.choices[0].message, "content", None), "strip", lambda: None)()
            if not content:
                logging.warning("LLM returned empty response")
                return "N/A"

            return content

        except Exception as e:
            logging.exception(f"LLM generation error: {e}")
            return "N/A"

    async def generate_async(self, prompt: str, timeout: int = 60) -> str:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.generate, prompt),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logging.warning("LLM generation timed out")
            return "N/A"