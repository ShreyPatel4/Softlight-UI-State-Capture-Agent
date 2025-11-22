import asyncio

from src.agent.browser import BrowserSession
from src.config import settings


async def main() -> None:
    async with BrowserSession() as session:
        # Open a neutral page for manual navigation
        await session.goto("https://www.google.com")
        # Keep the browser open so the user can log into Linear/Notion/LinkedIn manually.
        print("Browser started with persistent profile.")
        print("Log into Linear / Notion / LinkedIn in this window, then close it when done.")
        # Sleep forever until user closes browser
        while True:
            await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
