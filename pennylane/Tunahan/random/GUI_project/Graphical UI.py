import customtkinter as ctk
from quantum_ml_logic import QuantumMlAlgorithm

# Create the main window
app = ctk.CTk()
app.iconbitmap("assets/MONGO.ico")
app.title("Quantum Circuit Training")
app.geometry("900x540")

alg = QuantumMlAlgorithm()

# Create a custom font
customFont = ("Helvetica", 12)
app.option_add("*Font", customFont)

# Create a text widget for logs
log_text = ctk.CTkTextbox(app, width=100, height=10)

# Function to handle single run button click
def handle_single_run_button_click():
    # Call the function for a single run
    starting_params = alg.guess_starting_params()
    trained_params = alg.train_circuit(starting_params, alg.training_iterations)
    alg.evaluate_circuit(trained_params)

# Function to handle loop run button click
def handle_loop_run_button_click():
    # Create an entry field for loop count
    loop_entry = ctk.CTkEntry(app)
    loop_entry.pack()

    # Create an entry field for save option
    save_entry = ctk.CTkEntry(app)
    save_entry.pack()

    # Create an entry field for filename
    filename_entry = ctk.CTkEntry(app)
    filename_entry.pack()

    # Get user input from entry fields
    user_input_loop = loop_entry.get()
    user_input_save = save_entry.get()
    user_input_filename = filename_entry.get()

    # Clear the entry fields
    loop_entry.delete(0, ctk.END)
    save_entry.delete(0, ctk.END)
    filename_entry.delete(0, ctk.END)

    # Call the function for a loop
    for i in range(int(user_input_loop)):
        starting_params = alg.guess_starting_params()
        trained_params = alg.train_circuit(starting_params, alg.training_iterations)
        alg.evaluate_circuit(trained_params)

        if user_input_save.lower() == 'y' or user_input_save.lower() == 'yes':
            user_input_filename_counted = user_input_filename + "_loop_" + str(i)
            alg.save_params(trained_params, user_input_filename_counted,
                            path='Saved Circuit Models/Discrete Loops')

# Function to handle variable loop run button click
def handle_variable_loop_run_button_click():
    # Create an entry field for training iterations
    training_iterations_entry = ctk.CTkEntry(app)
    training_iterations_entry.pack()

    # Create an entry field for save option
    save_entry = ctk.CTkEntry(app)
    save_entry.pack()

    # Create an entry field for filename
    filename_entry = ctk.CTkEntry(app)
    filename_entry.pack()

    # Get user input from entry fields
    training_iterations_str = training_iterations_entry.get()
    if not training_iterations_str.isdigit():
        print("Invalid input", "Please enter a valid number for training iterations.")
        return
    training_iterations = int(training_iterations_str)
    user_input_save = save_entry.get()
    user_input_filename = filename_entry.get()

    # Clear the entry fields
    training_iterations_entry.delete(0, ctk.END)
    save_entry.delete(0, ctk.END)
    filename_entry.delete(0, ctk.END)

    if user_input_save.lower() == 'y' or user_input_save.lower() == 'yes':
        starting_params = alg.guess_starting_params()
        trained_params = alg.train_circuit(starting_params, training_iterations)
        alg.evaluate_circuit(trained_params)
        alg.save_params(trained_params, user_input_filename)

    else:
        starting_params = alg.guess_starting_params()
        trained_params = alg.train_circuit(starting_params, training_iterations)
        alg.evaluate_circuit(trained_params)


# Create a heading for the app
heading = ctk.CTk(app, text="Quantum Circuit Training", font=("Helvetica", 24))
heading.place(relx=0.5, rely=0.1, anchor="center")

# Create a button for single run
single_run_button = ctk.CTkButton(master=app, text="Single Run", command=handle_single_run_button_click)
single_run_button.place(relx=0.5, rely=0.2, anchor="center")

# Create a button for loop run
loop_run_button = ctk.CTkButton(master=app, text="Loop Run", command=handle_loop_run_button_click)
loop_run_button.place(relx=0.5, rely=0.3, anchor="center")

variable_loop_run_button = ctk.CTkButton(master=app, text="Variable Loop Run", command=handle_variable_loop_run_button_click)
variable_loop_run_button.place(relx=0.5, rely=0.4, anchor="center")


# Create a button for training
button = ctk.CTkButton(master=app, text="Start Training",
                       command=lambda: print("Start Training", "Start Training"),
                       corner_radius=15)
button.place(relx=0.5, rely=0.5, anchor="center")
#button.pack(padx=10, pady=10)

# Start the main loop
app.mainloop()