from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Base Strategy class per assignment §7.
    Strategies implement: on_start, on_tick, on_bar, on_fill, on_end
    """
    def __init__(self):
        pass

    @abstractmethod
    def on_start(self):
        """Called when the backtest starts."""
        pass

    @abstractmethod
    def on_tick(self, tick):
        """Called for every market data tick."""
        pass

    @abstractmethod
    def on_bar(self, bar):
        """Called on bar completion (e.g., 1-min, 5-min bars)."""
        pass

    @abstractmethod
    def on_fill(self, fill):
        """Called when an order is filled."""
        pass

    @abstractmethod
    def on_end(self):
        """Called when the backtest ends."""
        pass


