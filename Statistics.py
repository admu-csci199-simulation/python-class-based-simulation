from typing import List, Tuple, Dict
from collections import defaultdict
from PostInteraction import PostInteraction
import matplotlib.pyplot as plt
import numpy as np
import Agent
import os


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

    def aggregateCountsByBin(counts, binSize):
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


    def countInteractionPerPost(self) -> dict:
        """
        Counts how many interactions each post receives at each time step.
        Builds nested counts: counts[time][post] = numberOfInteractions
        """

        counts = defaultdict(lambda: defaultdict(int)) # defaultdict automatically creates missing keys when you access them
        for interaction in self._gatherInteractions():
            counts[interaction.time][interaction.post] += 1

        # revert to a normal dictionary
        countsDict = {}
        for time, post in counts.items():
            countsDict[time] = dict(post)

        return countsDict
    
    def generateGraphsPerPost(
        self,
        figsize: Tuple[int, int] = (8, 5),
        saveDir: str = None,
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
            raise ValueError("No interactions to plot.")

        posts = sorted({interaction.post for interaction in interactions})
        if not posts:
            raise ValueError("No posts found in interactions.")
        
        figs: Dict[str, Tuple[plt.Digure, plt.Axes]] = {}
        for post in posts:
            fig, ax = plt.subplots(figsize=figsize)

            postCounts = []
            for t in times:
                timeCounts = counts.get(t, {}) # Get the dictiornary of posts at time t
                count = timeCounts.get(post, 0) 
                postCounts.append(count)

            ax.bar(times, postCounts)
            ax.set_xticks(times)
            ax.set_xticklabels(times)

            ax.set_xlabel("Time")
            ax.set_ylabel("Count of Interactions")
            ax.set_title(f"Interactions for post: {post}")
            fig.tight_layout()

            if saveDir:
                os.makedirs(saveDir, exist_ok=True)
                filename = f"{saveDir}/post_{post}.png"
                fig.savefig(filename)

            figs[post] = (fig, ax)

        return figs