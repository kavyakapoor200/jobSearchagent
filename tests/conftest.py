import asyncio
import sys

# On Windows Python 3.11, ProactorEventLoop can trigger 0xC0000005 on interpreter teardown
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
