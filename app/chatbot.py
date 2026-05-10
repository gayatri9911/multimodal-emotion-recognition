def chatbot_response(emotion):

    responses = {

        "happy":
        "You sound happy today! Keep smiling.",

        "sad":
        "I am here for you. Things will get better.",

        "angry":
        "Take a deep breath and relax.",

        "fear":
        "Don't worry. You are stronger than you think.",

        "neutral":
        "You seem calm and balanced.",

        "calm":
        "You sound peaceful today.",

        "disgust":
        "Something seems to bother you.",

        "surprise":
        "That sounds unexpected!"
    }

    return responses.get(
        emotion,
        "I am listening."
    )