from tkinter import *

root = Tk()

root.title("AI Chatbot")

def send_query():

    send_query = "You -> "+e.get()

    txt.insert(END, "\n"+send_query)

    user_name = e.get().lower()

    if(user_name == "hello"):

        txt.insert(END, "\n" + "Bot -> Hi")

    elif(user_name == "hi" or user_name == "hai" or user_name == "hiiii"):

        txt.insert(END, "\n" + "Bot -> Hello")

    elif(e.get() == "How are you doing?"):

        txt.insert(END, "\n" + "Bot -> I’m fine and what about you")

    elif(user_name == "fine" or user_name == "I am great" or user_name == "I am doing good"):

        txt.insert(END, "\n" + "Bot -> Amazing! how can I help you.")

    else:

        txt.insert(END, "\n" + "Bot -> Sorry! I did not get you")

    e.delete(0, END)

txt = Text(root)

txt.grid(row=0, column=0, columnspan=2)

e = Entry(root, width=100)

e.grid(row=1, column=0)

send_query = Button(root, text="Send", command=send_query).grid(row=1, column=1)

root.mainloop()