from typing import List, Tuple, Dict
from collections import defaultdict
from PostInteraction import PostInteraction
import matplotlib.pyplot as plt
import numpy as np
import os
import Agent
import Constants
import imageio


class Statistics:
    def __init__(self, agents: "list[Agent.Agent]"):
        """
        Parameters:
        - interactions: list of PostInteractions
        """
        self.agents = agents


    def _gatherInteractions(self) -> List[PostInteraction]:
        """
        Flatten all agents' interactions into a single list of PostInteraction objects.
        """
        allInteractions = []
        for agent in self.agents:
            interactions = getattr(agent, "interactionsDone", [])
            for interaction in interactions:
                allInteractions.append(interaction)
        return allInteractions   


    def countInteractionPerPost(self) -> dict:
        """
        Counts how many interactions each post receives at each time step.
        Builds nested counts: counts[time][post] = numberOfInteractions
        """
        counts = defaultdict(lambda: defaultdict(int))
        for interaction in self._gatherInteractions():
            counts[interaction.time][interaction.post] += 1

        # revert to a normal dictionary
        countsDict = {}
        for time, post in counts.items():
            countsDict[time] = dict(post)

        return countsDict

    def generatePerTickGraphs(
        self,
        figsize: Tuple[int, int] = Constants.FIG_SIZE,
        saveDir: str = Constants.GRAPHS_DIR,
        postsSubset: List[str] = None
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """
        Create a bar chart per post.

        Parameters:
        - figsize: tuple for figure size (width, height)
        - saveDir: if provided, saves each figure as '{save_dir}/post_{post}.png
        - postsSubset: optional list of posts to plot (default: all posts).

        Returns:
        - dict mapping post -> (fig, ax)
        """
        counts = self.countInteractionPerPost()
        interactions = self._gatherInteractions()
        times = sorted(counts.keys())
        if not times:
            print("No interactions to plot.")
        posts = sorted({interaction.post for interaction in interactions}, key=lambda post: post.postID)
        print([post.beliefValue for post in posts])
        if not posts:
            print("No posts found in interactions.")
        
        figs: Dict[str, Tuple[plt.Digure, plt.Axes]] = {}
        os.makedirs(saveDir, exist_ok=True) 
        for post in posts:
            fig, ax = plt.subplots(figsize=Constants.FIG_SIZE)

            postCounts = []
            for t in times:
                timeCounts = counts.get(t, {}) # Get the dictiornary of posts at time t
                count = timeCounts.get(post, 0) 
                postCounts.append(count)

            ax.bar(times, postCounts)
            step = max(1, len(times) // 15)
            ax.set_xticks(times[::step])
            ax.set_xticklabels(times[::step], rotation=45)

            ax.set_xlabel("Time")
            ax.set_ylabel("Count of Interactions")
            ax.set_title(f"Interactions for post: {post}")
            
            fig.tight_layout()
            fig.savefig(f"{saveDir}/tick_post_{post}.png")

            plt.close(fig)

    def generatePerTickGraphsWithBelief(
        self,
        figsize: Tuple[int, int] = Constants.FIG_SIZE,
        saveDir: str = Constants.GRAPHS_DIR,
        postsSubset: List[str] = None
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """
            Parameters:
            - figsize: tuple for figure size (width, height)
            - saveDir: if provided, saves each figure as '{save_dir}/post_{post}.png
            - postsSubset: optional list of posts to plot (default: all posts).
        """
        counts = self.countInteractionPerPost()
        interactions = self._gatherInteractions()

        times_from_counts = set(counts.keys()) if counts else set()
        times_from_interactions = set()
        for interaction in interactions:
            t = interaction.time
            times_from_interactions.add(t)
        times = sorted(times_from_counts.union(times_from_interactions))

        posts_all = sorted(
            {interaction.post for interaction in interactions},
            key=lambda post: post.postID
        )

        if postsSubset is not None:
            posts = sorted(set(postsSubset))
        else:
            posts = posts_all

        category_counts: Dict[object, Dict[object, List[int]]] = {}
        for t in times:
            category_counts[t] = {post: [0, 0, 0] for post in posts}

        for interaction in interactions:
            t = interaction.time
            post = interaction.post
            belief = interaction.agentBelief

            belief_category = None  

            if -4 <= belief <= -2:
                belief_category = 0
            elif -1 <= belief <= 1:
                belief_category = 1
            elif 2 <= belief <= 4:
                belief_category = 2

            category_counts[t][post][belief_category] += 1

        times = sorted(times)

        figs: Dict[str, Tuple[plt.Figure, plt.Axes]] = {}

        colors = ["#A9221B", "#858585", "#07618f"]
        labels = ["Red", "Centrist", "Blue"]

        for post in posts:
            fig, ax = plt.subplots(figsize=figsize)
            cat0 = []
            cat1 = []
            cat2 = []
            for t in times:
                post_counts = category_counts.get(t, {}).get(post, [0, 0, 0])
                cat0.append(post_counts[0])
                cat1.append(post_counts[1])
                cat2.append(post_counts[2])

            x = list(times)

            width = None
            try:
                numeric_x = [float(v) for v in x]
                diffs = [j - i for i, j in zip(numeric_x[:-1], numeric_x[1:]) if (j - i) > 0]
                if diffs:
                    width = min(diffs) * 0.8
            except Exception:
                width = None

            if width is not None:
                p0 = ax.bar(x, cat0, width=width, color=colors[0], label=labels[0])
                p1 = ax.bar(x, cat1, bottom=cat0, width=width, color=colors[1], label=labels[1])
                bottom_cat0_cat1 = [a + b for a, b in zip(cat0, cat1)]
                p2 = ax.bar(x, cat2, bottom=bottom_cat0_cat1, width=width, color=colors[2], label=labels[2])
            else:
                p0 = ax.bar(x, cat0, color=colors[0], label=labels[0])
                p1 = ax.bar(x, cat1, bottom=cat0, color=colors[1], label=labels[1])
                bottom_cat0_cat1 = [a + b for a, b in zip(cat0, cat1)]
                p2 = ax.bar(x, cat2, bottom=bottom_cat0_cat1, color=colors[2], label=labels[2])

            step = max(1, len(times) // 15)
            xtick_positions = x[::step]
            xtick_labels = [str(t) for t in xtick_positions]
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, rotation=45)

            ax.set_xlabel("Time")
            ax.set_ylabel("Count of Interactions")
            ax.set_title(f"Interactions for post: {post} (Stacked by Belief Category)")
            ax.legend()

            fig.tight_layout()
            fig_path = os.path.join(saveDir, f"stacked_tick_post_{post}.png")
            fig.savefig(fig_path)

            figs[post] = (fig, ax)

        return figs

    def aggregateCountsByBin(self, counts, binSize):
        """
        Groups minute-by-minute interaction counts into bins.

        Parameters:
        - counts: dict[time -> dict[post -> count]]
        - bin_size: number of minutes per bin (e.g., 15 for 15-minute bins)

        Returns:
        - (binned_times, binned_counts)
            binned_times: sorted list of bin start times
            binned_counts: dict[bin_start -> dict[post -> total_count_in_bin]]
        """
        binned = defaultdict(lambda: defaultdict(int))
        for time, posts in counts.items():
            binStart =  (time // binSize) * binSize  
            for post, count in posts.items():
                binned[binStart][post] += count

        binnedTimes = sorted(binned.keys())

        # Convert from default dictionary to normal dictionary
        binnedCounts = {bt: dict(posts) for bt, posts in binned.items()}
        return binnedTimes, binnedCounts

    def generateBinnedGraphs(self, binSize=Constants.BIN_SIZE, figsize=Constants.FIG_SIZE, saveDir=Constants.GRAPHS_DIR):
        """
        Creates one bar chart per post, aggregating interactions into time bins.

        Parameters:
        - bin_size: how many minutes per bin 
        - figsize: (width, height) in inches for each chart
        - save_dir: path to save all charts as PNGs
        """
        counts = self.countInteractionPerPost()
        timesBinned, countsBinned = self.aggregateCountsByBin(counts, binSize)

        interactions = self._gatherInteractions()
        allPosts = sorted({i.post for i in interactions}, key=lambda post: post.postID)

        if not allPosts:
            print("No posts found to plot.")
        
        os.makedirs(saveDir, exist_ok=True)
        for post in allPosts:
            postCounts = []
            for bt in timesBinned:
                binData = countsBinned.get(bt, {})
                count = binData.get(post, 0)
                postCounts.append(count)

            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(timesBinned, postCounts, width=binSize*0.9)

            ax.set_xlabel(f"Time (binned every {binSize} mins)")
            ax.set_ylabel("Interaction Count")
            ax.set_title(f"Interactions for Post: {post}")
            ax.set_xticks(timesBinned[::max(1, len(timesBinned)//10)])  # Show only ~10 ticks
            ax.set_xticklabels(timesBinned[::max(1, len(timesBinned)//10)], rotation=45)
            
            fig.tight_layout()
            fig.savefig(f"{saveDir}/binned_post_{post}.png")

            plt.close(fig)

    def generateAgentDemogGraph(self, figsize=Constants.FIG_SIZE, saveDir=Constants.GRAPHS_DIR):
        """
        Creates a pie chart of agent types.
        """
        counts = {"gullible": 0, "normal": 0, "stubborn": 0}
        for agent in self.agents:
            type = agent.classifyAgentType()
            counts[type] += 1

        labels = list(counts.keys())
        values = list(counts.values())
        total = sum(values)

        # Custom function to show both % and counts
        def autopct_format(pct):
            count = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({count})"

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct=autopct_format,
            startangle=140,
            textprops={"fontsize": 10}
        )

        ax.set_title(f"Response Type Distribution")
        ax.axis("equal") 

        os.makedirs(saveDir, exist_ok=True)
        fig.savefig(f"{saveDir}/response_type_distribution.png", bbox_inches="tight")
        plt.close(fig)

        return fig

    def generateBeliefTypePieChart(self, saveDir=Constants.GRAPHS_DIR, filename='default.png', addLabels=True):
        """
            Generates a pie chart showing the distribution of agents
            based on their beliefValue (red, centrist, blue).
        """
        counts = {"red": 0, "centrist": 0, "blue": 0}
        for agent in self.agents:
            type = agent.classifyAgentBelief()
            counts[type] += 1

        
        labels = list(counts.keys()) if addLabels else None
        sizes = list(counts.values())
        total = sum(sizes)

        def autopct_format(pct):
            count = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({count})"

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=autopct_format,
            startangle=140,
            textprops={"fontsize": 10},
            colors=["#e74c3c", "#95a5a6", "#3498db"]  # red, gray, blue
        )

        ax.set_title(f"Agent Belief Distribution")
        ax.axis("equal")

        os.makedirs(saveDir, exist_ok=True)
        fig.savefig(f"{saveDir}/{filename}", bbox_inches="tight")
        plt.close(fig)

        return fig

    def generateGifBeliefType(
        self,
        sourceFolder=Constants.GIF_FRAMES_DIR,
        outputPath=Constants.GIF_OUT_PATH,
        duration=Constants.GIF_DURATION,  # normal per-frame duration (seconds)
        buffer_seconds=10                 # freeze on first & last frame
    ):
        """
        Creates a GIF from PNG images in sourceFolder using imageio.
        Adds:
            - infinite looping
            - 3-second buffer on the first and last frame
        """

        os.makedirs(sourceFolder, exist_ok=True)
        files = [f for f in os.listdir(sourceFolder) if f.endswith(".png")]

        if not files:
            print("No PNG images found in folder:", sourceFolder)
            return

        files.sort()  # ensure animation order
        print(files)

        # Load images
        frames = []
        for filename in files:
            filepath = os.path.join(sourceFolder, filename)
            frames.append(imageio.imread(filepath))

        # Build per-frame durations
        durations = []
        for i in range(len(frames)):
            if i == 0 or i == len(frames) - 1:
                durations.append(buffer_seconds)   # hold for buffer
            else:
                durations.append(duration)         # normal duration

        # Save GIF (imageio loops forever by default)
        imageio.mimsave(
            outputPath,
            frames,
            duration=durations,
            loop=True
        )

        print(f"GIF saved to: {outputPath}")


    def generateAgentTypesPerCamp(self, figsize=Constants.FIG_SIZE, saveDir=Constants.GRAPHS_DIR):
        """
        Generate one pie chart per political camp (red, centrist, blue).
        Each pie shows distribution of agent types inside that camp:
          gullible, normal, stubborn, unknown

        Saves files to saveDir
        """

        # initialize nested counters: camp -> type -> count
        camps = ["red", "centrist", "blue"]
        types = ["gullible", "normal", "stubborn"]
        camp_counts = {camp: {t: 0 for t in types} for camp in camps}

        # fill counts
        for agent in self.agents:
            camp = agent.classifyAgentBelief()
            a_type = agent.classifyAgentType()

            if camp not in camp_counts:
                # ignore agents whose belief isn't one of the three camps
                continue
            camp_counts[camp][a_type] += 1

        saved_paths = {}
        # create a pie per camp
        for camp in camps:
            counts = camp_counts[camp]
            labels = list(counts.keys())
            values = list(counts.values())
            total = sum(values) if sum(values) > 0 else 1

            def autopct_format(pct):
                count = int(round(pct * total / 100.0))
                return f"{pct:.1f}%\n({count})"

            fig, ax = plt.subplots(figsize=figsize)
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct=autopct_format,
                startangle=140,
                textprops={"fontsize": 10}
            )
            ax.set_title(f"{camp.capitalize()} camp's response type composition")
            ax.axis("equal")

            fname = os.path.join(saveDir, f"{(camp)}_camp_response_types.png")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            saved_paths[camp] = fname

    def generateActiveStatusChart(self, figsize=Constants.FIG_SIZE, saveDir=Constants.GRAPHS_DIR):
        """
        Generates a bar chart showing online vs offline agents at each time step.
        Positive bars (above y=0) are online counts, negative bars (below y=0) are offline counts.
        """
        # Count online agents at each time step (using the same step as before)
        onlineCountPerTime = defaultdict(int)
        sample_times = list(range(0, Constants.MAXIMUM_TIME, 15))
        for currentTime in sample_times:
            for agent in self.agents:
                if agent.isOnline(currentTime):
                    onlineCountPerTime[currentTime] += 1

        times = sorted(onlineCountPerTime.keys())
        if not times:
            print("No time data to plot.")
            return

        online_counts = [onlineCountPerTime[t] for t in times]
        total_agents = max(1, len(self.agents))
        offline_counts = [total_agents - o for o in online_counts]

        x = np.array(times)
        # choose a reasonable bar width based on time spacing
        if len(times) > 1:
            spacing = np.min(np.diff(x))
            width = spacing * 0.8
        else:
            width = 10.0

        # positive for online, negative for offline
        heights_online = online_counts
        heights_offline = [-v for v in offline_counts]

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x, heights_online, width=width, color='#2ecc71', label='Online')
        ax.bar(x, heights_offline, width=width, color='#e74c3c', label='Offline')

        ax.axhline(0, color='black', linewidth=0.8)

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Agent count (positive = online, negative = offline)")
        ax.set_title("Online vs Offline Agent Count Over Time")
        step = max(1, len(times) // 15)
        ax.set_xticks(times[::step])
        ax.set_xticklabels(times[::step], rotation=45)
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)

        # symmetric y-limits so zero is centered visually
        max_count = max(max(online_counts), max(offline_counts), total_agents)
        ax.set_ylim(-max_count * 1.05, max_count * 1.05)

        fig.tight_layout()
        os.makedirs(saveDir, exist_ok=True)
        fig.savefig(f"{saveDir}/active_status_chart.png")
        plt.close(fig)

        return fig

    def getLayersPerPosts(self):
        interactions = self._gatherInteractions()
        postsLayers = {} # postID : {nth Layer : [agentID]}

        for interaction in interactions:
            if not interaction.isShared:
                continue
            post_id = interaction.post.postID
            agent = interaction.agent
            nth_layer = interaction.layer

            if post_id not in postsLayers:
                postsLayers[post_id] = {}
            
            if nth_layer not in postsLayers[post_id]:
                postsLayers[post_id][nth_layer] = []

            postsLayers[post_id][nth_layer].append(agent)

        # TESTING CODE
        # for post_id, data in postsLayers.items():
        #     print(f'Post ID: {post_id}')
        #     totalAgents = 0
        #     for i in range(min(data.keys()), max(data.keys()) + 1):
        #         if i not in data:
        #             print(f"no Layer {i} in Post {post_id}.")
        #             continue
        #         print(f'\tLayer: {i} has {len(data[i])} agents.')
        #         totalAgents += len(data[i])
        #     print(f"Post {post_id} was at least seen by {totalAgents} agents.")
        return postsLayers


    def generateGraphs(self, saveDir=''):
        os.makedirs(saveDir, exist_ok=True)
        self.generateBeliefTypePieChart(saveDir=saveDir)
        self.generatePerTickGraphs(saveDir=saveDir)
        self.generateBinnedGraphs(saveDir=saveDir)
        self.generateAgentDemogGraph(saveDir=saveDir)
        self.generateBeliefTypePieChart(saveDir=saveDir, filename='agent_belief_distribution.png')
        self.generateAgentTypesPerCamp(saveDir=saveDir)
        self.generateActiveStatusChart(saveDir=saveDir)
        self.generatePerTickGraphsWithBelief(saveDir=saveDir)
        self.generateGifBeliefType()
        
        # for tests
        self.getLayersPerPosts()