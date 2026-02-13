import customtkinter as ctk
import json
import random
from pathlib import Path


class PostFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.posts = app.posts  # shared reference

        # Grid config (for centering)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # =======================
        # Title
        # =======================
        title = ctk.CTkLabel(
            self,
            text="Post Interface",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, pady=(20, 25))

        # =======================
        # Manual Inputs
        # =======================
        manual_label = ctk.CTkLabel(
            self,
            text="Manual Inputs",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        manual_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=40)

        self.post_id = self.create_entry("Post ID", 2)
        self.post_topic = self.create_entry("Post Topic", 3)
        self.belief_value = self.create_entry("Belief Value", 4)
        self.interest_value = self.create_entry("Interest Value", 5)
        self.posting_time = self.create_entry("Posting Time", 6)

        self.add_btn = ctk.CTkButton(
            self,
            text="Add Post",
            command=self.add_post
        )
        self.add_btn.grid(
            row=7,
            column=0,
            columnspan=2,
            pady=(15, 25),
            sticky="n"
        )

        # =======================
        # Random Generation
        # =======================
        random_label = ctk.CTkLabel(
            self,
            text="Random Generation",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        random_label.grid(row=8, column=0, columnspan=2, sticky="w", padx=40)

        self.random_count_label = ctk.CTkLabel(
            self,
            text="Number of Random Posts"
        )
        self.random_count_label.grid(
            row=9, column=0, padx=(40, 10), pady=8, sticky="w"
        )

        self.random_count_entry = ctk.CTkEntry(self, width=220)
        self.random_count_entry.grid(
            row=9, column=1, padx=(10, 40), pady=8, sticky="w"
        )

        self.random_btn = ctk.CTkButton(
            self,
            text="Generate Random Posts",
            command=self.generate_random_posts
        )
        self.random_btn.grid(
            row=10,
            column=0,
            columnspan=2,
            pady=(10, 30),
            sticky="n"
        )

        # =======================
        # Save JSON (FINAL ACTION)
        # =======================
        self.save_btn = ctk.CTkButton(
            self,
            text="Save to JSON",
            command=self.save_json
        )
        self.save_btn.grid(
            row=11,
            column=0,
            columnspan=2,
            pady=(10, 10),
            sticky="n"
        )

        self.status = ctk.CTkLabel(self, text="")
        self.status.grid(
            row=12,
            column=0,
            columnspan=2,
            pady=(5, 20)
        )

    # =======================
    # Helpers
    # =======================
    def create_entry(self, label_text, row):
        label = ctk.CTkLabel(self, text=label_text)
        label.grid(
            row=row,
            column=0,
            padx=(40, 10),
            pady=8,
            sticky="w"
        )

        entry = ctk.CTkEntry(self, width=220)
        entry.grid(
            row=row,
            column=1,
            padx=(10, 40),
            pady=8,
            sticky="w"
        )

        return entry

    # =======================
    # Manual Post Logic
    # =======================
    def add_post(self):
        try:
            post = {
                "postID": int(self.post_id.get()),
                "postTopic": int(self.post_topic.get()),
                "beliefValue": int(self.belief_value.get()),
                "interestValue": int(self.interest_value.get()),
                "postingTime": int(self.posting_time.get())
            }

            self.posts.append(post)

            self.status.configure(
                text=f"Added post #{post['postID']}",
                text_color="green"
            )

            for entry in [
                self.post_id,
                self.post_topic,
                self.belief_value,
                self.interest_value,
                self.posting_time
            ]:
                entry.delete(0, "end")

        except ValueError:
            self.status.configure(
                text="All manual fields must be integers",
                text_color="red"
            )

    # =======================
    # Random Generation Logic
    # =======================
    def generate_random_posts(self):
        try:
            count = int(self.random_count_entry.get())
            if count <= 0:
                raise ValueError

            start_id = self.posts[-1]["postID"] + 1 if self.posts else 0

            for i in range(count):
                post = {
                    "postID": start_id + i,
                    "postTopic": random.randint(1, 5),
                    "beliefValue": random.randint(-4, 4),
                    "interestValue": random.randint(-4, 4),
                    "postingTime": random.randint(0, 2880)
                }
                self.posts.append(post)

            self.status.configure(
                text=f"Generated {count} random posts",
                text_color="green"
            )

            self.random_count_entry.delete(0, "end")

        except ValueError:
            self.status.configure(
                text="Enter a valid positive integer",
                text_color="red"
            )

    # =======================
    # Save JSON
    # =======================
    def save_json(self):
        output = {
            "Agents": self.app.agent_data,
            "Posts": self.posts
        }

        Path("output").mkdir(exist_ok=True)

        with open("output/config.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)

        self.status.configure(
            text="Saved to output/config.json",
            text_color="green"
        )
