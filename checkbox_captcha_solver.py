from solvers.base_solver import BaseSolver
from utils.logger import setup_logger
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = setup_logger("CheckboxCaptchaSolver")

class CheckboxCaptchaSolver(BaseSolver):
    """
    Solver for simple "I am not a robot" check‑the‑box CAPTCHAs.
    Assumes `browser` is a Selenium WebDriver instance.
    """
    def __init__(self, model, config):
        super().__init__(model, config)
        # model not used here, but config may contain timeouts
        self.timeout = config.get("timeout", 30)

    def solve(self, browser) -> bool:
        """
        Finds the reCAPTCHA iframe, clicks the checkbox, and waits
        for the "checked" state or callback token to appear.
        Returns True if it succeeded.
        """
        try:
            logger.info("Locating reCAPTCHA checkbox iframe")
            # 1) Find the iframe that contains the checkbox
            iframe = WebDriverWait(browser, self.timeout).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "iframe[src*='recaptcha']")))
            browser.switch_to.frame(iframe)

            # 2) Click the actual checkbox
            checkbox = WebDriverWait(browser, self.timeout).until(
                EC.element_to_be_clickable((By.ID, "recaptcha-anchor")))
            checkbox.click()
            logger.info("Clicked checkbox — waiting for verification")

            # 3) Back to main content, wait for the token/input to appear
            browser.switch_to.default_content()
            WebDriverWait(browser, self.timeout).until(
                EC.text_to_be_present_in_element_value(
                    (By.CSS_SELECTOR, "textarea[g-recaptcha-response]"),
                    ""  # it will become non-empty
                )
            )
            logger.info("reCAPTCHA token injected — checkbox solved")
            return True

        except Exception as e:
            logger.error(f"Checkbox CAPTCHA solve failed: {e}")
            return False
