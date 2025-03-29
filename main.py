# main.py
import cv2
from vision_agent import VisionAgent
from reasoning_agent import ReasoningAgent
from action_agent import ActionAgent

# Initialize agents
vision_agent = VisionAgent()
reasoning_agent = ReasoningAgent(use_llm=False)  # Set to True if you have a local LLM
action_agent = ActionAgent()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break

    # Vision: Detect objects and text
    objects, text = vision_agent.detect(frame)

    # Reasoning: Generate description
    description = reasoning_agent.generate_description(objects, text)
    print("Description:", description)

    # Action: Speak the description
    action_agent.speak(description)

    # Display the frame (optional)
    cv2.imshow("VisionMate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()