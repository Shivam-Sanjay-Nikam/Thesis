# Define global variables for managing state
mode = "wait_circle"  # Modes: wait_circle, storing_a, storing_b, operation
a = 0
b = 0
current_number = 0
gesture_history = []  # Store all gestures for debugging
result = None  # Store the result of operations

def process_gesture(gesture):
    """Process gestures for a gesture-controlled calculator in real time."""
    global mode, a, b, current_number, gesture_history, result

    # Trim any leading or trailing spaces from the gesture
    gesture = gesture.strip()
    print(f"Received gesture: {gesture}")

    # Append the gesture to history
    gesture_history.append(gesture)
    print(f"Gesture history: {gesture_history}")

    # Print the current mode and relevant state
    print(f"Current mode: {mode}")
    print(f"Current 'a': {a}, Current 'b': {b}, Current Number: {current_number}, Result: {result}")

    # Handle gestures based on the current mode
    if gesture == "Circle":
        if mode == "wait_circle":
            print("Circle detected: Starting to store value for 'a'")
            mode = "storing_a"
            current_number = 0
        elif mode == "storing_a":
            print(f"Circle detected: Stored 'a' as {current_number}")
            a = current_number
            print("Circle detected: Starting to store value for 'b'")
            mode = "storing_b"
            current_number = 0
        elif mode == "storing_b":
            print(f"Circle detected: Stored 'b' as {current_number}")
            b = current_number
            print("Circle detected: Ready for operation gestures")
            mode = "operation"
        elif mode == "operation":
            print("Circle detected: But in operation mode")
            mode = "operation"

    elif mode in ["storing_a", "storing_b"]:
        if gesture == "Swipe Right":
            print("Swipe Right detected: Incrementing current number by 1")
            current_number += 1
        elif gesture == "Swipe Left":
            print("Swipe Left detected: Decrementing current number by 1")
            current_number -= 1

    elif mode == "operation":
        if gesture == "Swipe Up":
            result = a + b
            print(f"Operation detected: {a} + {b} = {result}")
        elif gesture == "Swipe Down":
            result = a - b
            print(f"Operation detected: {a} - {b} = {result}")
        elif gesture == "Swipe Right":
            result = a * b
            print(f"Operation detected: {a} * {b} = {result}")
        elif gesture == "Swipe Left":
            if b != 0:
                result = a / b
                print(f"Operation detected: {a} / {b} = {result}")
            else:
                result = None
                print("Error detected: Division by zero is not allowed")
        # Reset to wait for the next sequence
        print("Setting to wait for the next operation...")
        mode = "operation"

    # Print updated state
    print(f"Updated mode: {mode}")
