import speech_recognition as sr
from ai_model import PowerControlAIModel

def recognize_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your command...")
        audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio)
            print("You said:", command)
            return command
        except sr.UnknownValueError:
            print("Could not understand the command.")
            return None
        except sr.RequestError:
            print("Could not request results; check your internet connection.")
            return None

def main():
    # Step 1: Load the trained AI model
    model = PowerControlAIModel()
    model.load_model(model_dir="./saved_model")

    # Step 2: Recognize voice command
    command = recognize_voice_command()

    if command:
        # Step 3: Predict the intent
        predicted_intent = model.predict(command)
        print(f"Predicted Intent: {predicted_intent}")

if __name__ == "__main__":
    main()
