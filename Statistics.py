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
            raise ValueError("No interactions to plot.")

        posts = sorted({interaction.post for interaction in interactions})
        if not posts:
            raise ValueError("No posts found in interactions.")
        
        figs: Dict[str, Tuple[plt.Digure, plt.Axes]] = {}
        os.makedirs(saveDir, exist_ok=True) 
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
            raise ValueError("No posts found to plot.")
        
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