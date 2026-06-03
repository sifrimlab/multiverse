import os
import time

from playwright.sync_api import expect, sync_playwright


def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the Streamlit app
        try:
            page.goto("http://localhost:8502")
            # Wait for the app to load
            page.wait_for_selector("text=Multiverse Setup Wizard", timeout=20000)

            # Fill out the form
            page.get_by_label("Dataset Path (h5ad/h5mu)").fill("data/test_dataset.h5mu")
            page.get_by_label("Batch Key").fill("batch_id")
            page.get_by_label("Cell Type Key (Optional)").fill("cell_type")

            # Select models (it's a multiselect, let's just use what's there or try to click)
            # Default should have 'pca' selected. Let's try to add 'mofa'
            page.get_by_role("combobox", name="Select Models to Run").click()
            page.get_by_text("mofa", exact=True).click()

            # Click submit
            page.get_by_role("button", name="Generate Configuration").click()

            # Wait for success message
            expect(
                page.get_by_text("Configuration saved to generated_config.json!")
            ).to_be_visible()

            # Take screenshot
            verification_dir = os.path.join(os.getcwd(), "verification")
            os.makedirs(verification_dir, exist_ok=True)
            page.screenshot(path=os.path.join(verification_dir, "gui_verification.png"))
            print("Verification successful, screenshot saved.")

        except Exception as e:
            print(f"Verification failed: {e}")
            # Try to take a screenshot of the failure state
            verification_dir = os.path.join(os.getcwd(), "verification")
            os.makedirs(verification_dir, exist_ok=True)
            page.screenshot(path=os.path.join(verification_dir, "gui_failure.png"))
        finally:
            browser.close()


if __name__ == "__main__":
    run_verification()
