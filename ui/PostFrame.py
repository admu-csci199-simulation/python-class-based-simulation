import customtkinter as ctk
import json
import random
from pathlib import Path
from ui.SimulateFrame import SimulateFrame

class PostFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.posts = app.posts  # shared reference
        self._original_interest_value = None

        # Grid config (for centering)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.back_btn = ctk.CTkButton(
            self,
            text="← Back",
            width=90,
            command=self.go_back
        )

        self.back_btn.grid(
            row=0,
            column=0,
            padx=20,
            pady=(15, 5),
            sticky="w"
        )

        title = ctk.CTkLabel(
            self,
            text="Post Interface",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title.grid(row=1, column=0, columnspan=2, pady=(20, 25))

        manual_label = ctk.CTkLabel(
            self,
            text="Manual Inputs",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        manual_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=40)

        self.post_topic_label, self.post_topic = self.create_entry("Post Topic", 3, "Integer value from 1 to 5")
        self.belief_label, self.belief_value = self.create_entry("Belief Value", 4, "Integer value from -4 to 4")
        self.interest_label, self.interest_value = self.create_entry("Interest Value", 5, "Integer value from -10 to 21")
        self.posting_time_label, self.posting_time = self.create_entry("Posting Time", 6, "Integer value from 0 to 2880")
         
        # Boolean variable
        self.misinformation_var = ctk.BooleanVar(value=False)

        # Label
        misinfo_label = ctk.CTkLabel(self, text="Misinformation?")
        misinfo_label.grid(
            row=7,
            column=0,
            padx=(40, 10),
            pady=8,
            sticky="w"
        )

        # Checkbox
        self.misinformation_checkbox = ctk.CTkCheckBox(
            self,
            text="",
            variable=self.misinformation_var,
            command=self.toggle_misinformation_bonus
        )
        self.misinformation_checkbox.grid(
            row=7,
            column=1,
            padx=(10, 40),
            pady=8,
            sticky="w"
        )

        self.add_btn = ctk.CTkButton(
            self,
            text="Add Post",
            command=self.add_post
        )
        self.add_btn.grid(
            row=8,
            column=0,
            columnspan=2,
            pady=(15, 25),
            sticky="n"
        )

        random_label = ctk.CTkLabel(
            self,
            text="Random Generation",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        random_label.grid(row=9, column=0, columnspan=2, sticky="w", padx=40)

        self.random_count_label = ctk.CTkLabel(
            self,
            text="Number of Random Posts"
        )
        self.random_count_label.grid(
            row=10, column=0, padx=(40, 10), pady=8, sticky="w"
        )

        self.random_count_entry = ctk.CTkEntry(self, width=220)
        self.random_count_entry.grid(
            row=10, column=1, padx=(10, 40), pady=8, sticky="w"
        )

        self.random_btn = ctk.CTkButton(
            self,
            text="Generate",
            command=self.generate_random_posts
        )
        self.random_btn.grid(
            row=11,
            column=0,
            columnspan=2,
            pady=(10, 30),
            sticky="n"
        )

        self.save_btn = ctk.CTkButton(
            self,
            text="Save to JSON",
            command=self.save_json
        )
        self.save_btn.grid(
            row=12,
            column=0,
            columnspan=2,
            pady=(10, 10),
            sticky="n"
        )

        self.status = ctk.CTkLabel(self, text="")
        self.status.grid(
            row=13,
            column=0,
            columnspan=2,
            pady=(5, 20)
        )

    # Helpers
    def create_entry(self, label_text, row, placeholder=""):
        label = ctk.CTkLabel(self, text=label_text)
        label.grid(
            row=row,
            column=0,
            padx=(40, 10),
            pady=8,
            sticky="w"
        )

        entry = ctk.CTkEntry(
            self,
            width=220,
            placeholder_text=placeholder
        )
        entry.grid(
            row=row,
            column=1,
            padx=(10, 40),
            pady=8,
            sticky="w"
        )

        return label, entry
    
    def go_back(self):
        self.app.show_agent_screen()


    def add_post(self):
        try:
            base_interest = int(self.interest_value.get())
            is_misinfo = self.misinformation_var.get()

            posts_added = 0

            # Apply +3 interest for OG misinformation
            interest_val = base_interest + 3 if is_misinfo else base_interest


            # Create original post
            post_id = len(self.posts)

            original_post = {
                "postID": post_id,
                "postTopic": int(self.post_topic.get()),
                "beliefValue": int(self.belief_value.get()),
                "interestValue": interest_val,
                "postingTime": int(self.posting_time.get()),
                "misinformation": is_misinfo,
                "spawn": is_misinfo  # Only OG misinfo can spawn
            }

            self.posts.append(original_post)
            posts_added += 1


            # Spawn 2 additional posts (ONLY if OG misinfo)
            if original_post["misinformation"] and original_post["spawn"]:

                previous_interest = interest_val
                previous_time = original_post["postingTime"]

                for _ in range(2):

                    post_id = len(self.posts)

                    # Progressive interest growth
                    new_interest = previous_interest + 3

                    # Sequential 5–12 hour delay
                    delay = random.randint(300, 720)
                    new_time = previous_time + delay

                    spawned_post = {
                        "postID": post_id,
                        "postTopic": original_post["postTopic"],
                        "beliefValue": original_post["beliefValue"],
                        "interestValue": new_interest,
                        "postingTime": new_time,
                        "misinformation": True,
                        "spawn": False  # Spawned posts cannot spawn again
                    }

                    self.posts.append(spawned_post)

                    previous_interest = new_interest
                    previous_time = new_time
                    posts_added += 1


            # Status Message
            if posts_added == 3:
                self.status.configure(
                    text="3 posts have been added",
                    text_color="green"
                )
            else:
                self.status.configure(
                    text="1 post has been added",
                    text_color="green"
                )


            # Clear fields
            for entry in [
                self.post_topic,
                self.belief_value,
                self.interest_value,
                self.posting_time
            ]:
                entry.delete(0, "end")

            self.misinformation_var.set(False)

        except ValueError:
            self.status.configure(
                text="All manual fields must be integers",
                text_color="red"
            )

    def generate_random_posts(self):
        try:
            count = int(self.random_count_entry.get())
            if count <= 0:
                raise ValueError

            posts_added = 0

            for _ in range(count):
                is_misinfo = random.choice([True, False])

                base_interest = random.randint(-4, 4)
                interest_val = base_interest + 3 if is_misinfo else base_interest

                post_id = len(self.posts)

                original_post = {
                    "postID": post_id,
                    "postTopic": random.randint(1, 5),
                    "beliefValue": random.randint(-4, 4),
                    "interestValue": interest_val,
                    "postingTime": random.randint(0, 2880),
                    "misinformation": is_misinfo,
                    "spawn": is_misinfo
                }

                self.posts.append(original_post)
                posts_added += 1


                # Spawn logic for OG misinformation
                if original_post["misinformation"] and original_post["spawn"]:

                    previous_interest = interest_val
                    previous_time = original_post["postingTime"]

                    for _ in range(2):

                        post_id = len(self.posts)

                        new_interest = previous_interest + 3
                        delay = random.randint(300, 720)
                        new_time = previous_time + delay

                        spawned_post = {
                            "postID": post_id,
                            "postTopic": original_post["postTopic"],
                            "beliefValue": original_post["beliefValue"],
                            "interestValue": new_interest,
                            "postingTime": new_time,
                            "misinformation": True,
                            "spawn": False
                        }

                        self.posts.append(spawned_post)

                        previous_interest = new_interest
                        previous_time = new_time
                        posts_added += 1

                self.status.configure(
                    text=f"{posts_added} posts have been added",
                    text_color="green"
                )

                self.random_count_entry.delete(0, "end")

        except ValueError:
            self.status.configure(
                text="Enter a valid positive integer",
                text_color="red"
            )

    def toggle_misinformation_bonus(self):
        current_text = self.interest_value.get().strip()

        # Checkbox turned ON
        if self.misinformation_var.get():

            # Update label
            self.interest_label.configure(
                text="Interest Value (+3)",
                text_color="red"
            )

            if current_text != "":
                try:
                    current_value = int(current_text)

                    # Store original only once
                    self._original_interest_value = current_value

                    boosted_value = current_value + 3

                    self.interest_value.delete(0, "end")
                    self.interest_value.insert(0, str(boosted_value))

                except ValueError:
                    pass

        # Checkbox turned OFF
        else:

            self.interest_label.configure(
                text="Interest Value",
                text_color="white"
            )

            # Restore original value if we have one
            if self._original_interest_value is not None:
                self.interest_value.delete(0, "end")
                self.interest_value.insert(0, str(self._original_interest_value))

            self._original_interest_value = None

    def save_json(self):
        output = {
            "Agents": self.app.agent_data,
            "Posts": self.posts
        }

        Path("input").mkdir(exist_ok=True)

        with open("input/config.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)

        # Redirect to SimulateFrame
        self.app.clear_frame()
        self.app.current_frame = SimulateFrame(self.app, self.app)
        self.app.current_frame.pack(padx=20, pady=20)
        self.app.auto_resize()