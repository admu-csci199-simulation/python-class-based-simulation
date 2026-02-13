from ui.PostFrame import PostFrame
from ui.AgentFrame import AgentFrame
from ui.PostFrame import PostFrame
from ui.StartFrame import StartFrame
from tkinter import filedialog

import customtkinter as ctk
import json
import random

class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Agent Simulation Setup")

        self.agent_data = {}
        self.posts = []

        self.current_frame = None

        self.show_start_screen()
        self.resizable(False, False)


    # ------------------------
    # Frame Switching
    # ------------------------

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



    # ------------------------
    # JSON Upload Flow
    # ------------------------

    def start_json_upload_flow(self):

        # Step 1 — Upload Agents.json
        agent_path = filedialog.askopenfilename(
            title="Select Agents JSON",
            filetypes=[("JSON Files", "*.json")]
        )

        if not agent_path:
            return

        try:
            with open(agent_path, "r", encoding="utf-8") as f:
                self.agent_data = json.load(f)
        except Exception as e:
            print("Agent JSON error:", e)
            return

        # Step 2 — Upload Posts.json
        post_path = filedialog.askopenfilename(
            title="Select Posts JSON",
            filetypes=[("JSON Files", "*.json")]
        )

        if not post_path:
            return

        try:
            with open(post_path, "r", encoding="utf-8") as f:
                self.posts = json.load(f)
        except Exception as e:
            print("Post JSON error:", e)
            return

        # Go directly to Post screen
        self.show_post_screen(self.agent_data)

    def auto_resize(self):
            self.update_idletasks()
            width = self.winfo_reqwidth()
            height = self.winfo_reqheight()
            self.geometry(f"{width}x{height}")


def run():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    run()