# action_agent.py
import pyttsx3

class ActionAgent:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

    def speak(self, description):
        self.engine.say(description)
        self.engine.runAndWait()

if __name__ == "__main__":
    action_agent = ActionAgent()
    description = "I see a person at x=100, y=200, and some text that says 'Exit sign'."
    action_agent.speak(description)