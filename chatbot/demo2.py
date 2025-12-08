import nltk

from nltk.chat.util import Chat, reflections

dialogues = [

    [

        r"my name is (.*)",

        ["Hello %1, How are you?",]

    ],

    [

        r"hi|hey|hello",

        ["Hello", "Hey",]

    ], 

    [

        r"what is your name ?",

        ["I am a bot created by Dev713. You can call me Sentenial X!",]

    ],

    [

        r"how are you ?",

        ["I'm doing good, How about you?",]

    ],

    [

        r"sorry (.*)",

        ["It's alright","Its ok, never mind",]

    ],

    [

        r"I am great",

        ["Glad to hear that, How can I assist you?",]

    ],

    [

        r"i'm (.*) doing good",

        ["Great to hear that","How can I help you?:)",]

    ],

    [

        r"(.*) age?",

        ["I'm a chatbot, bro. \nI do not have an age.",]

    ],

    [

        r"what (.*) want ?",

        ["Provide me an offer I cannot refuse",]

    ],

    [

        r"(.*) created?",

        ["DEV713 created me using Python's NLTK library ","It’s a top secret ;)",]

    ],

    [

        r"(.*) (location|city) ?",

        ['Odisha, Bhubaneswar',]

    ],

    [

        r"how is the weather in (.*)?",

        ["Weather in %1 is awesome as always","It’s too hot in %1","It’s too cold in %1","I do not know much about %1"]

    ],

    [

        r"i work in (.*)?",

        ["%1 is a great company; I have heard that they are in huge loss these days.",]

    ],

    [

        r"(.*)raining in (.*)",

        ["There is no rain since last week in %2","Oh, it's raining too much in %2"]

    ],

    [

        r"how (.*) health(.*)",

        ["I'm a chatbot, so I'm always healthy ",]

    ],

    [

        r"(.*) (sports|game) ?",

        ["I'm a huge fan of cricket",]

    ],

    [

        r"who (.*) sportsperson ?",

        ["Dhoni","Jadeja","AB de Villiars"]

    ],

    [

        r"who (.*) (moviestar|actor)?",

        ["Tom Cruise"]

    ],

    [

        r"I am looking for online tutorials and courses to learn data science and artificial-intelligence. Can you suggest some?",

        ["Analytics Drift has several articles offering clear, step-by-step guides with code examples for quick, practical learning in data science and AI."]

    ],

    [

        r"quit",

        ["Goodbye, see you soon.","It was nice talking to you. Bye."]

    ],

]

def chatbot():

    print("Hi! I am a chatbot built by Analytics Drift for your service")

    chatbot = Chat(dialogues, reflections)

    chatbot.converse()

if __name__ == "__main__":

    chatbot()
