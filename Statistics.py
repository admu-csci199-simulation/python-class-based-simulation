from typing import List, Tuple, Dict
from collections import defaultdict
from PostInteraction import PostInteraction
import matplotlib.pyplot as plt
import numpy as np
import os
import Agent
import Constants


class Statistics:
    def __init__(self, agents: List[Agent.Agent]):
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

        posts = sorted({interaction.post for interaction in interactions})
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
        allPosts = sorted({i.post for i in interactions})

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
        counts = {"gullible": 0, "normal": 0, "stubborn": 0, "unknown": 0}
        for agent in self.agents:
            type = agent.classifyAgentType()
            if type not in counts:
                counts["unknown"] += 1
            else:
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

        ax.set_title(f"Agent Type Distribution")
        ax.axis("equal") 

        fig.savefig(f"{saveDir}/agent_type_distribution.png", bbox_inches="tight")
        plt.close(fig)

        return fig

    def generateBeliefTypePieChart(self, saveDir=Constants.GRAPHS_DIR):
        """
            Generates a pie chart showing the distribution of agents
            based on their beliefValue (red, centrist, blue).
        """
        counts = {"red": 0, "centrist": 0, "blue": 0, "unknown": 0}
        for agent in self.agents:
            type = agent.classifyAgentBelief()
            if type not in counts:
                counts["unknown"] += 1
            else:
                counts[type] += 1

        labels = list(counts.keys())
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

        fig.savefig(f"{saveDir}/_agent_belief_distribution.png", bbox_inches="tight")
        plt.close(fig)

        return fig
    
    def generateAgentTypesPerCamp(self, figsize=Constants.FIG_SIZE, saveDir=Constants.GRAPHS_DIR):
        """
        Generate one pie chart per political camp (red, centrist, blue).
        Each pie shows distribution of agent types inside that camp:
          gullible, normal, stubborn, unknown

        Saves files to saveDir
        """

        # initialize nested counters: camp -> type -> count
        camps = ["red", "centrist", "blue"]
        types = ["gullible", "normal", "stubborn", "unknown"]
        camp_counts = {camp: {t: 0 for t in types} for camp in camps}

        # fill counts
        for agent in self.agents:
            try:
                camp = agent.classifyAgentBelief()
            except Exception:
                camp = "unknown"
            try:
                a_type = agent.classifyAgentType()
            except Exception:
                a_type = "unknown"

            if camp not in camp_counts:
                # ignore agents whose belief isn't one of the three camps
                continue

            if a_type not in camp_counts[camp]:
                a_type = "unknown"
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
            ax.set_title(f"{camp.capitalize()} camp — agent type composition")
            ax.axis("equal")

            fname = os.path.join(saveDir, f"{(camp)}_camp_agent_types.png")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            saved_paths[camp] = fname

    def generateActiveStatusChart(self, figsize=Constants.FIG_SIZE, saveDir=Constants.GRAPHS_DIR):
        """
        Generates a line chart showing the count of online agents at each time step.
        """
        # Count online agents at each time step
        onlineCountPerTime = defaultdict(int)
        
        for currentTime in range(0, Constants.MAXIMUM_TIME, 30):
            for agent in self.agents:
                if agent.isOnline(currentTime):
                    onlineCountPerTime[currentTime] += 1
        
        times = sorted(onlineCountPerTime.keys())
        counts = [onlineCountPerTime[t] for t in times]
        
        if not times:
            print("No time data to plot.")
            return
        
        # Create the line chart
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, counts, linewidth=2, color='#3498db', marker='o', markersize=3)
        
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Number of Online Agents")
        ax.set_title("Online Agent Count Over Time")
        ax.set_xticks(times[::max(1, len(times)//15)])
        ax.set_xticklabels(times[::max(1, len(times)//15)], rotation=45)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        os.makedirs(saveDir, exist_ok=True)
        fig.savefig(f"{saveDir}/active_status_chart.png")
        plt.close(fig)
        
        return fig

    def generateGraphs(self, saveDir=''):
        os.makedirs(saveDir, exist_ok=True)
        self.generateBeliefTypePieChart(saveDir=saveDir)
        self.generatePerTickGraphs(saveDir=saveDir)
        self.generateBinnedGraphs(saveDir=saveDir)
        self.generateAgentDemogGraph(saveDir=saveDir)
        self.generateBeliefTypePieChart(saveDir=saveDir)
        self.generateAgentTypesPerCamp(saveDir=saveDir)
        self.generateActiveStatusChart(saveDir=saveDir)