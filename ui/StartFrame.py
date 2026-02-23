import customtkinter as ctk


class StartFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        ctk.CTkLabel(
            self,
            text="Choose Setup",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=10)

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack()

        manual_btn = ctk.CTkButton(
            button_frame,
            text="Manual Setup",
            width=200,
            command=self.app.show_agent_screen
        )
        manual_btn.grid(row=0, column=0, padx=20)

        json_btn = ctk.CTkButton(
            button_frame,
            text="JSON Upload",
            width=200,
            command=self.app.start_json_upload_flow
        )
        json_btn.grid(row=0, column=1, padx=20)
