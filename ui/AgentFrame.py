import customtkinter as ctk
import json
import random

class AgentFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # Back
        back_btn = ctk.CTkButton(
            self,
            text="← Back",
            width=75,
            command=self.go_back
        )
        back_btn.pack(anchor="w", padx=20, pady=(15, 5))

        # Title
        ctk.CTkLabel(
            self,
            text="Agent Interface",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(10, 20))
        
        self.entries = {}
        self.create_entry("Agent Count", callback=self.update_share_propensity)
        self.create_section("Response Types")
        self.create_entry("Gullible Count")
        self.create_entry("Normal Count")
        self.create_entry("Stubborn Count")

        self.create_section("Share Propensity Types (Auto Computed)")

        self.lurker_var = ctk.StringVar(value="0")
        self.normal_var = ctk.StringVar(value="0")
        self.active_var = ctk.StringVar(value="0")

        self.create_display("Lurker Count (90%)", self.lurker_var)
        self.create_display("Normal Sharer Count (9%)", self.normal_var)
        self.create_display("Active Count (1%)", self.active_var)

        self.create_section("Belief Type")
        self.create_entry("Red Count")
        self.create_entry("Centrist Count")
        self.create_entry("Blue Count")

        # Button container
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=20)

        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        next_btn = ctk.CTkButton(
            button_frame,
            text="Next",
            command=self.validate_and_continue,
            width=180
        )
        next_btn.grid(row=0, column=1, padx=10)


        self.status = ctk.CTkLabel(self, text="")
        self.status.pack()

    def create_section(self, title):
        ctk.CTkLabel(
            self, text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=40, pady=(20, 5))

    def create_entry(self, label, callback=None):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=40, pady=4)

        ctk.CTkLabel(frame, text=label).pack(side="left")

        entry = ctk.CTkEntry(frame, width=120)
        entry.pack(side="right")

        self.entries[label] = entry

        if callback:
            entry.bind("<KeyRelease>", lambda e: callback())

    def validate_and_continue(self):
        try:
            data = {k: int(v.get()) for k, v in self.entries.items()}

            # Auto-computed share propensity
            data["Lurker Count"] = int(self.lurker_var.get())
            data["Normal Share Count"] = int(self.normal_var.get())
            data["Active Count"] = int(self.active_var.get())

            agent_count = data["Agent Count"]

            def check_total(keys):
                return sum(data[k] for k in keys) == agent_count

            # Agent type distribution
            if not check_total(["Gullible Count", "Normal Count", "Stubborn Count"]):
                raise ValueError("Type counts do not sum to Agent Count")

            # Share propensity distribution
            if not check_total(["Lurker Count", "Normal Share Count", "Active Count"]):
                raise ValueError("Share Propensity counts do not sum to Agent Count")

            # Belief distribution
            if not check_total(["Red Count", "Centrist Count", "Blue Count"]):
                raise ValueError("Belief counts do not sum to Agent Count")

            self.app.show_post_screen(data)

        except ValueError as e:
            self.status.configure(text=str(e), text_color="red")


    def upload_json(self):
        file_path = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON Files", "*.json")]
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Expecting structure:
            # {
            #     "Agents": {...},
            #     "Posts": [...]
            # }

            agent_data = data.get("Agents")
            posts = data.get("Posts", [])

            if not agent_data:
                raise ValueError("Invalid JSON format: Missing 'Agents'")

            # Fill UI fields
            for key, entry in self.entries.items():
                if key in agent_data:
                    entry.delete(0, "end")
                    entry.insert(0, str(agent_data[key]))

            # Store posts into app memory
            self.app.posts = posts

            self.status.configure(
                text="JSON loaded successfully",
                text_color="green"
            )

        except Exception as e:
            self.status.configure(
                text=f"Error loading JSON: {e}",
                text_color="red"
            )

    def create_display(self, label, variable):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=40, pady=4)

        ctk.CTkLabel(frame, text=label).pack(side="left")

        entry = ctk.CTkEntry(frame, width=120, textvariable=variable)
        entry.pack(side="right")

        entry.configure(state="disabled")

    def update_share_propensity(self):
        try:
            agent_count = int(self.entries["Agent Count"].get())

            lurker = round(agent_count * 0.90)
            normal = round(agent_count * 0.09)
            active = agent_count - lurker - normal

            self.lurker_var.set(str(lurker))
            self.normal_var.set(str(normal))
            self.active_var.set(str(active))

        except ValueError:
            self.lurker_var.set("0")
            self.normal_var.set("0")
            self.active_var.set("0")

    def go_back(self):
        self.app.show_start_screen()