"""
Auto mouse scroll script - Scroll up/down every 10 minutes
Prevents screen lock and keeps system active
"""

import pyautogui
import time
from datetime import datetime

# Settings
SCROLL_INTERVAL = 600  # 10 minutes = 600 seconds
SCROLL_AMOUNT = 300    # Scroll distance
SCROLL_UP_DOWN_DELAY = 1  # Delay between up/down scroll (seconds)

def scroll_gesture():
    """Execute up/down alternating scroll"""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Executing mouse scroll...")

        # Scroll down
        pyautogui.scroll(-SCROLL_AMOUNT)
        print(f"  -> Scrolled DOWN {SCROLL_AMOUNT} units")

        time.sleep(SCROLL_UP_DOWN_DELAY)

        # Scroll up
        pyautogui.scroll(SCROLL_AMOUNT)
        print(f"  -> Scrolled UP {SCROLL_AMOUNT} units")

        print(f"  -> Scroll complete")

    except Exception as e:
        print(f"  -> Error: {e}")

def main():
    """Main program - execute scroll every 10 minutes"""
    print("=" * 60)
    print("Auto Mouse Scroll Program Started")
    print(f"Scroll interval: {SCROLL_INTERVAL}s ({SCROLL_INTERVAL/60} minutes)")
    print(f"Scroll amount: +/-{SCROLL_AMOUNT} units")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    try:
        while True:
            # Execute scroll
            scroll_gesture()

            # Wait for next execution
            print(f"\nWaiting {SCROLL_INTERVAL/60} minutes...\n")
            time.sleep(SCROLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nProgram stopped")
        print("=" * 60)

if __name__ == "__main__":
    main()
