import cv2
from vision_agent import VisionAgent
from reasoning_agent import ReasoningAgent
from action_agent import ActionAgent

# Initialize agents
vision_agent = VisionAgent()
reasoning_agent = ReasoningAgent(use_llm=False)  # Set to True if you have a local LLM
action_agent = ActionAgent()

print("VisionMate is running... Press 'q' to exit, '2' to toggle OCR mode.")
# Open webcam
cap = cv2.VideoCapture(0)

# Mode flag: False = Object Detection only, True = Object Detection + OCR
ocr_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break

    # Vision: Detect objects (and text if in OCR mode)
    objects, text = vision_agent.detect(frame)
    
    # Depending on mode, include or exclude text in the description
    if ocr_mode:
        description = reasoning_agent.generate_description(objects, text)
    else:
        description = reasoning_agent.generate_description(objects, "")  # No text in default mode
    
    print("Description:", description)

    # Action: Speak the description
    action_agent.speak(description)

    # Display the frame (optional)
    cv2.imshow("VisionMate", frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('2'):
        ocr_mode = not ocr_mode  # Toggle OCR mode
        mode_status = "OCR Mode ON" if ocr_mode else "OCR Mode OFF"
        print(mode_status)
        action_agent.speak(mode_status)  # Optional: Announce mode change

cap.release()
cv2.destroyAllWindows()