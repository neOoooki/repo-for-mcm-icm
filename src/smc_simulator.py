import pandas as pd
import numpy as np
from scipy.stats import dirichlet

class FanVoteSimulator:
    def __init__(self, data_path, num_particles=1000):
        self.df = pd.read_csv(data_path)
        self.num_particles = num_particles
        self.results = []
        
        # Hyperparameters for state transition
        self.inertia_weight = 0.7  # How much previous popularity matters
        self.performance_weight = 0.3 # How much current judge score matters
        self.noise_level = 5.0 # Concentration parameter for Dirichlet (higher = less noise)

    def run_simulation(self):
        seasons = self.df['season'].unique()
        all_estimated_votes = []

        for season in seasons:
            print(f"Simulating Season {season}...")
            season_data = self.df[self.df['season'] == season]
            season_votes = self._simulate_season(season_data, season)
            all_estimated_votes.extend(season_votes)
            
        return pd.DataFrame(all_estimated_votes)

    def _simulate_season(self, season_data, season):
        weeks = sorted(season_data['week'].unique())
        contestants = season_data['celebrity_name'].unique()
        num_contestants = len(contestants)
        
        # Initialize particles: [num_particles, num_contestants]
        # Initial distribution is uniform-ish Dirichlet
        particles = np.random.dirichlet(np.ones(num_contestants), self.num_particles)
        
        season_results = []
        
        # Map contestant name to index for this season
        c_to_idx = {name: i for i, name in enumerate(contestants)}
        idx_to_c = {i: name for i, name in enumerate(contestants)}

        prev_particles = particles.copy()

        for week in weeks:
            week_data = season_data[season_data['week'] == week]
            
            # Identify who is still in the competition this week
            active_contestants = week_data['celebrity_name'].values
            if len(active_contestants) == 0:
                continue
                
            active_indices = [c_to_idx[c] for c in active_contestants]
            
            # --- 1. State Transition (Prediction Step) ---
            # FanVote_t ~ Dirichlet( alpha * FanVote_{t-1} + beta * JudgeScore_t )
            
            # Get normalized judge scores for active contestants
            judge_scores = week_data.set_index('celebrity_name')['total_judge_score']
            # Handle case where all scores are 0 or NaN
            if judge_scores.sum() == 0:
                 normalized_scores = np.ones(len(active_indices)) / len(active_indices)
            else:
                 normalized_scores = judge_scores.values / judge_scores.sum()
            
            # Update particles
            new_particles = np.zeros((self.num_particles, num_contestants))
            
            for i in range(self.num_particles):
                # Get previous votes for ACTIVE contestants only
                prev_active = prev_particles[i, active_indices]
                # Re-normalize previous votes so they sum to 1 among active contestants
                if prev_active.sum() > 0:
                    prev_active /= prev_active.sum()
                else:
                    prev_active = np.ones(len(active_indices)) / len(active_indices)
                
                # Mean of the Dirichlet proposal
                # Combination of history (inertia) and current performance
                alpha_mean = (self.inertia_weight * prev_active) + \
                             (self.performance_weight * normalized_scores)
                
                # Scale by noise level to get alpha parameters
                # Add small epsilon to avoid zero
                alphas = alpha_mean * self.noise_level + 1e-6
                
                # Sample new votes for this week
                new_votes_active = np.random.dirichlet(alphas)
                
                # Assign back to full array (inactive ones stay 0)
                new_particles[i, active_indices] = new_votes_active

            # --- 2. Update/Resampling (Correction Step) ---
            # Check consistency with elimination result
            
            # Find the actual eliminated contestant this week
            eliminated_row = week_data[week_data['is_eliminated'] == 1]
            eliminated_name = eliminated_row['celebrity_name'].iloc[0] if not eliminated_row.empty else None
            
            weights = np.zeros(self.num_particles)
            
            for i in range(self.num_particles):
                # Calculate total combined score for this particle
                # Rules vary by season, but here we implement a generic Rank-based method for S1-2
                # and Percentage-based for others as a simplification, 
                # or strictly check if the eliminated person is at the bottom.
                
                # Simplified check: Calculate total score/rank and see if actual eliminated is lowest
                
                # We need to simulate the specific season's rule. 
                # Let's use a "Rank Sum" approach as it's common and robust.
                # Rank of Judge Score + Rank of Fan Vote. Lowest Sum = Eliminated.
                # Note: In DWTS, high score is good, so Rank 1 = Best. 
                # Lowest combined score is actually Worst.
                # Wait, usually "Lowest Score" is eliminated.
                # Let's standardize: Higher Value = Better.
                
                p_votes = new_particles[i, active_indices]
                p_judge = judge_scores.values # Already extracted above
                
                # Rank them (higher rank number = better, so rank 1 is worst)
                # scipy rankdata: rank 1 is smallest value.
                from scipy.stats import rankdata
                
                # Judge Ranks (1 = Lowest Score)
                r_judge = rankdata(p_judge, method='min') 
                # Fan Ranks (1 = Lowest Vote)
                r_fan = rankdata(p_votes, method='min')
                
                # Combined Rank
                # In early seasons: Rank + Rank.
                # In later seasons: Percentage + Percentage.
                
                # To be robust, we accept the particle if the actual eliminated contestant
                # is in the Bottom 2 of the combined ranking.
                # This covers both strict elimination and "Judges Save" (S28+).
                
                total_metric = r_judge + r_fan # Higher is better
                
                # Identify who has the lowest metrics
                # We need the index in 'active_indices' corresponding to 'eliminated_name'
                
                if eliminated_name:
                    elim_idx_local = -1
                    for idx, name in enumerate(active_contestants):
                        if name == eliminated_name:
                            elim_idx_local = idx
                            break
                    
                    if elim_idx_local != -1:
                        # Check if this person is in bottom 2
                        # Get indices of two smallest values
                        bottom_2_indices = np.argsort(total_metric)[:2]
                        
                        if elim_idx_local in bottom_2_indices:
                            weights[i] = 1.0 # Valid particle
                        else:
                            weights[i] = 0.0 # Invalid particle
                    else:
                        # Actual eliminated person not found in data? (e.g. withdrawal)
                        # Assume all particles valid
                        weights[i] = 1.0
                else:
                    # No elimination this week (e.g. non-elimination round)
                    weights[i] = 1.0

            # --- Resample ---
            if weights.sum() == 0:
                # If all particles failed, relax constraints or keep previous (fallback)
                # This happens if parameters are off or data is weird.
                # For now, just re-use new_particles uniformly
                print(f"  Warning: Collapse at Season {season} Week {week}. All particles rejected.")
                weights = np.ones(self.num_particles) / self.num_particles
            else:
                weights /= weights.sum()
                
            # Systematic Resampling
            indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=weights)
            particles = new_particles[indices]
            
            # Store mean estimated votes for this week
            mean_votes = particles.mean(axis=0)
            std_votes = particles.std(axis=0) # Measure of certainty
            
            for idx, name in enumerate(contestants):
                if name in active_contestants:
                     # Find which local index this is
                     local_idx = list(active_contestants).index(name)
                     # Get the mean vote for this active contestant
                     # Note: particles are already filtered to only have active indices set
                     # But 'particles' shape is (N, Total_Contestants).
                     # The inactive cols are 0.
                     
                     vote_est = mean_votes[c_to_idx[name]]
                     certainty = std_votes[c_to_idx[name]]
                     
                     season_results.append({
                         'season': season,
                         'week': week,
                         'celebrity_name': name,
                         'estimated_fan_vote': vote_est,
                         'vote_certainty_std': certainty
                     })
            
            # Update prev_particles for next week
            prev_particles = particles.copy()

        return season_results

if __name__ == "__main__":
    import os
    input_csv = os.path.join("data", "processed_data.csv")
    simulator = FanVoteSimulator(input_csv, num_particles=1000)
    df_votes = simulator.run_simulation()
    
    output_path = os.path.join("data", "estimated_fan_votes.csv")
    df_votes.to_csv(output_path, index=False)
    print(f"Simulation complete. Saved to {output_path}")
    print(df_votes.head())
