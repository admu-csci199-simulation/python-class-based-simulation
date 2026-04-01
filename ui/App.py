from ui.PostFrame import PostFrame
from ui.AgentFrame import AgentFrame
from ui.PostFrame import PostFrame
from ui.StartFrame import StartFrame
from ui.SimulateFrame import SimulateFrame
from tkinter import filedialog

import customtkinter as ctk
import json
from Random import rng as random

class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Agent Simulation Setup")

        self.agent_data = {}
        self.posts = []

        self.current_frame = None

        self.show_start_screen()
        self.resizable(False, False)


    def clear_frame(self):
        if self.current_frame:
            self.current_frame.destroy()


    def show_start_screen(self):
        self.clear_frame()
        self.current_frame = StartFrame(self, self)
        self.current_frame.pack(padx=20, pady=20)

        self.auto_resize()


    def show_agent_screen(self):
        self.clear_frame()
        self.current_frame = AgentFrame(self, self)
        self.current_frame.pack(padx=20, pady=20)

        self.auto_resize()


    def show_post_screen(self, agent_data):
        self.agent_data = agent_data
        self.clear_frame()
        self.current_frame = PostFrame(self, self)
        self.current_frame.pack(padx=20, pady=20)

        self.auto_resize()


    def start_json_upload_flow(self):
        file_path = filedialog.askopenfilename(
            title="Select Config JSON",
            filetypes=[("JSON Files", "*.json")]
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate structure
            if "Agents" not in data or "Posts" not in data:
                raise ValueError("Invalid config file. Must contain 'Agents' and 'Posts'.")

            from pathlib import Path
            Path("input").mkdir(exist_ok=True)

            # Save to input/config.json
            with open("input/config.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            # Load into memory
            self.agent_data = data["Agents"]
            self.posts = data["Posts"]

            # Redirect to SimulateFrame
            self.clear_frame()
            self.current_frame = SimulateFrame(self, self)
            self.current_frame.pack(padx=20, pady=20)
            self.auto_resize()

        except Exception as e:
            print("Config JSON error:", e)

    def auto_resize(self):
        self.update_idletasks()

        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        self.geometry(f"{width}x{height}+{x}+{y}")


def run():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    run()