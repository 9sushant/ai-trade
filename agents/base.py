from abc import ABC, abstractmethod
from loguru import logger
import time


class BaseAgent(ABC):
    """Base class for all trading agents."""

    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.logger = logger.bind(agent=name)

    @abstractmethod
    def run_cycle(self) -> dict:
        """Execute one cycle of the agent's logic. Returns result dict."""

    def start(self, interval_seconds: int = 300):
        """Run agent in a loop at the given interval."""
        self.is_running = True
        self.logger.info(f"Agent [{self.name}] started (interval: {interval_seconds}s)")
        while self.is_running:
            try:
                result = self.run_cycle()
                self.on_cycle_complete(result)
            except Exception as e:
                self.logger.error(f"Agent [{self.name}] cycle error: {e}")
            time.sleep(interval_seconds)

    def stop(self):
        self.is_running = False
        self.logger.info(f"Agent [{self.name}] stopped")

    def on_cycle_complete(self, result: dict):
        """Hook called after each successful cycle."""
        pass
