import customtkinter as ctk


class SimulateFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            self,
            text="Ready to Simulate",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, pady=(40, 20))

        info = ctk.CTkLabel(
            self,
            text=f"Agents: {self.app.agent_data.get('Agent Count', 0)}\nPosts: {len(self.app.posts)}"
        )
        info.grid(row=1, column=0, columnspan=2, pady=(0, 30))

        back_btn = ctk.CTkButton(
            self,
            text="Back",
            width=75,
            command=self.go_back
        )
        back_btn.grid(row=2, column=0, pady=20)

        simulate_btn = ctk.CTkButton(
            self,
            text="Simulate",
            width=75,
            command=self.simulate
        )
        simulate_btn.grid(row=2, column=1, pady=20)

        # Status label (NEW)
        self.status = ctk.CTkLabel(self, text="")
        self.status.grid(row=3, column=0, columnspan=2, pady=(10, 20))

    def go_back(self):
        self.app.show_start_screen()

    def simulate(self):
        try:
            self.status.configure(text="Running simulation...", text_color="yellow")
            self.update_idletasks()

            from Main import runSimulation
            runSimulation()

            self.status.configure(
                text="Simulation completed successfully!",
                text_color="green"
            )

        except Exception as e:
            self.status.configure(
                text=f"Simulation failed: {str(e)}",
                text_color="red"
            )