"""
Standard lifecycle: runs the existing train() loop directly.
This is the default mode and preserves all existing Flotilla behavior.
"""


async def run(session_manager):
    await session_manager.train()
