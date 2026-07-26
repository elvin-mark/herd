import asyncio

from herd.services.manager import ProcessManager

# Global process manager
manager = ProcessManager()

# Global dictionary to track background downloads
pull_tasks = {}
pull_lock = asyncio.Lock()
