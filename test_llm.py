# test_llm.py
import asyncio
from app.core.llm_client import LLMClient

async def test():
    c = LLMClient()
    ok = await c.ping("/")
    print("ping ok:", ok)
    msgs = [{"role": "user", "content": "Say hello in one sentence."}]
    try:
        resp = await c.generate(messages=msgs, max_tokens=50, temperature=0.0)
        print("status:", resp.get("status_code"))
        print("model_used:", resp.get("model_used"))
        print("text:", resp.get("text") or str(resp.get("raw"))[:400])
    finally:
        await c.close()

if __name__ == "__main__":
    asyncio.run(test())
