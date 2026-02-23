import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import ttk
# from . import plot_gui, plot_gauss_newton
# from . import nlp_callback as cb


class GUI(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        # general settings
        self.protocol("WM_DELETE_WINDOW", self.quit_window)
        self.title('Optimal Control Solver')
        self.configure(bg="#2e2e2e")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background="#2e2e2e", foreground="white", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=5)
        style.configure("TCheckbutton", background="#2e2e2e", foreground="white")
        style.configure("TMenubutton", background="#3e3e3e", foreground="white",
                        font=("Segoe UI", 10))

        # ---------- Frames ----------
        self.left_frame = ttk.Frame(self, width=250)
        self.left_frame.grid(row=0, column=0, padx=15, pady=10, sticky="NS")

        self.right_frame = ttk.Frame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="NSEW")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.set_opt_entries()
        self.define_vars()
        self.insert_options()

        # ---------- Plot Area ----------
        self.fig = plt.figure(figsize=(15, 10), dpi=100)
        self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ---------- Run GUI ----------
        self.mainloop()

    def set_opt_entries(self):
        pass

    def define_vars(self):
        pass

    def insert_options(self):
        pass

    def solve(self):
        pass

    def plot(self):
        pass

    def adapt_init_type(self, sel_type):
        match sel_type:
            case "automatic":
                return "auto"
            case "random":
                return "rand"
            case "linear":
                return "lin"
            case _:
                return "auto"

    def stop_function(self):
        self.function_running.set(False)
        tk.Button(master=self.left_frame,
                  command=self.solve,
                  height=2,
                  width=10,
                  text="Solve").grid(row=self.last_row, column=0, sticky="NSEW", padx=5, pady=5)

    def start_function(self):
        self.function_running.set(True)
        tk.Button(master=self.left_frame,
                  command=self.stop_function,
                  height=2,
                  width=10,
                  text="Stop").grid(row=self.last_row, column=0, sticky="NSEW", padx=5, pady=5)

    def quit_window(self):
        self.quit()
        self.destroy()


