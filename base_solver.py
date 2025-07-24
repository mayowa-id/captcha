from abc import ABC, abstractmethod
from typing import Optional

class BaseSolver(ABC):
    """Base class for all CAPTCHA solvers."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    @abstractmethod
    def solve(self, image_path: str) -> Optional[str]:
        """
        Solve the CAPTCHA.
        
        Args:
            image_path (str): Path to the CAPTCHA image
            
        Returns:
            Optional[str]: The solved CAPTCHA text or None if failed
        """
        pass