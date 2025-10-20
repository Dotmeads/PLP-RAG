import random

# ---------------------------------------------------------------------------
# RESPONSE TEMPLATES
# ---------------------------------------------------------------------------

GREETING_RESPONSES = [
    "Hello again! 👋 How can I help you today?",
    "Hey there! 👋 Looking to adopt a pet or need some care tips?",
    "Hi! 👋 I can help you find your next furry friend or share pet care advice.",
    "Hello! 🐾 Want to look for a cat or dog to adopt?",
    "Heyy 👋 What kind of pet are you hoping to find today?"
]

THANKYOU_RESPONSES = [
    "You're most welcome! 😊 Anything else you'd like to ask?",
    "Happy to help! 🐾 Would you like to look for more pets?",
    "No problem at all 😄 I'm here whenever you need pet info!",
    "You're welcome! 🐶🐱 Want me to find more cute pets for you?",
]

GOODBYE_RESPONSES = [
    "Goodbye! 👋 Hope you find your perfect furry friend 🐶🐱",
    "See you later! 👋 Give your pets an extra cuddle for me 🐾",
    "Bye-bye! 👋 Wishing you and your pets lots of happiness!",
    "Take care! 🐾 Hope you'll visit again for more pet searches!",
]

FALLBACK_RESPONSES = [
    "I'm not sure I understood. 🤔 You can say 'I want to adopt a cat in Penang' or 'How to care for a puppy?'.",
    "Oops 😅 I didn't quite get that. Try saying 'I want to adopt a dog in Johor' or 'How to care for a kitten?'.",
    "Hmm 🤔 I might have missed that. You can ask me something like 'find a dog in KL' or 'care tips for cats'.",
]

# ---------------------------------------------------------------------------
# Unified response dictionary
# ---------------------------------------------------------------------------
RESPONSES = {
    "greeting": GREETING_RESPONSES,
    "thank_you": THANKYOU_RESPONSES,
    "goodbye": GOODBYE_RESPONSES,
}

# ---------------------------------------------------------------------------
# Helper: safely fetch a random response for an intent
# ---------------------------------------------------------------------------
def get_response(intent: str) -> str:
    """Returns a random response for known intents or a friendly fallback"""
    if intent in RESPONSES:
        return random.choice(RESPONSES[intent])
    else:
        return random.choice(FALLBACK_RESPONSES)
