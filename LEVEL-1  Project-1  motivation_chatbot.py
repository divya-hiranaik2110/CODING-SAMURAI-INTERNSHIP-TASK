import random
import datetime

# Predefined lists of motivational quotes and health tips
quotes = [
    "The only way to do great work is to love what you do.",
    "Success is not the key to happiness. Happiness is the key to success.",
    "Believe you can and you're halfway there.",
    "Hardships often prepare ordinary people for an extraordinary destiny."
]

health_tips = [
    "Drink plenty of water throughout the day.",
    "Take short breaks and stretch every hour.",
    "Get at least 7-8 hours of sleep for a healthy mind.",
    "Make time for a 30-minute walk each day."
]

# Function to greet the user based on time of day
def greeting():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good Morning! 😊"
    elif 12 <= hour < 18:
        return "Good Afternoon! ☀️"
    else:
        return "Good Evening! 🌙"

# Function to get the user's name
def get_user_name():
    name = input("What's your name? ")
    return name

# Function to give a random motivational quote
def give_motivation():
    return random.choice(quotes)

# Function to give a random health tip
def give_health_tip():
    return random.choice(health_tips)

# Main chatbot loop
def start_chatbot():
    print(greeting())
    name = get_user_name()
    print(f"Hello, {name}! How can I help you today?")
    
    while True:
        user_input = input("\nAsk me for motivation or health tips, or type 'exit' to end: ").lower()
        
        if 'motivation' in user_input:
            print(give_motivation())
        elif 'health' in user_input:
            print(give_health_tip())
        elif 'exit' in user_input:
            print("Goodbye! Stay motivated! 😊")
            break
        else:
            print("Sorry, I didn't understand that. Try asking for motivation or health tips.")

# Run the chatbot
if __name__ == "__main__":
    start_chatbot()
