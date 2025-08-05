import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Dict
import seaborn as sns

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GRPOExperimentPlotter:
    """GRPO experiment comparison plotter"""
    
    def __init__(self, experiments: List[Tuple[str, str, str]]):
        """
        Initialize plotter
        
        Args:
            experiments: List of experiments, each element is (experiment_name, training_log_path, validation_log_path)
        """
        self.experiments = experiments
        # 使用更清晰的颜色方案，避免黄色等不清晰的颜色
        # 可以选择以下几种方案之一：
        
        # 方案1: 使用 viridis 调色板（推荐，颜色对比度高）
        # self.colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
        
        # 方案2: 使用 plasma 调色板（蓝紫到橙黄的渐变，对比度好）
        # self.colors = plt.cm.plasma(np.linspace(0, 1, len(experiments)))
        
        # 方案3: 使用自定义颜色列表（手动选择清晰的颜色）
        # clear_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        #                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        # self.colors = clear_colors[:len(experiments)]
        
        # 方案4: 使用 tab10 但跳过黄色
        tab10_colors = plt.cm.tab10.colors
        # 移除黄色（索引1）和浅绿色（索引2），重新排列
        filtered_colors = [tab10_colors[0]] + list(tab10_colors[3:]) + [tab10_colors[2]]
        self.colors = np.array(filtered_colors[:len(experiments)])
        
    def load_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all experiment data"""
        data = {}
        
        for exp_name, train_file, val_file in self.experiments:
            print(f"Loading experiment data: {exp_name}")
            
            # Check if files exist
            if not os.path.exists(train_file):
                print(f"Warning: Training file does not exist - {train_file}")
                continue
            if not os.path.exists(val_file):
                print(f"Warning: Validation file does not exist - {val_file}")
                continue
            
            try:
                train_df = pd.read_csv(train_file)
                val_df = pd.read_csv(val_file)
                
                data[exp_name] = {
                    'train': train_df,
                    'val': val_df
                }
                print(f"Successfully loaded {exp_name}: {len(train_df)} training rows, {len(val_df)} validation rows")
                
            except Exception as e:
                print(f"Error loading experiment {exp_name} data: {e}")
                continue
        
        return data
    
    def plot_metrics_comparison(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                              save_path: str = "grpo_experiments_comparison.png"):
        """Plot comparison charts for six metrics"""
        
        metrics = [
            ('train_reward', 'Training Reward'),
            ('train_format_reward', 'Training Format Reward'),
            ('train_answer_reward', 'Training Answer Reward'),
            ('val_reward', 'Validation Reward'),
            ('val_format_reward', 'Validation Format Reward'),
            ('val_answer_reward', 'Validation Answer Reward')
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx]
            
            for i, (exp_name, exp_data) in enumerate(data.items()):
                if metric.startswith('train'):
                    df = exp_data['train']
                    x_col = 'step'
                else:
                    df = exp_data['val']
                    x_col = 'step'
                
                if metric in df.columns:
                    ax.plot(df[x_col], df[metric], 
                           label=exp_name, 
                           color=self.colors[i],
                           linewidth=2,
                           alpha=0.8)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Steps', fontsize=12)
            ax.set_ylabel('Reward Value', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Set y-axis range starting from 0
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison charts saved to: {save_path}")
        plt.show()
    
    def plot_combined_rewards(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                            save_path: str = "grpo_combined_rewards.png"):
        """Plot training and validation reward comparison"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training reward comparison
        for i, (exp_name, exp_data) in enumerate(data.items()):
            train_df = exp_data['train']
            ax1.plot(train_df['step'], train_df['train_reward'], 
                    label=f'{exp_name} (Train)', 
                    color=self.colors[i],
                    linewidth=2,
                    alpha=0.8)
        
        ax1.set_title('Training Reward Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Training Reward', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_ylim(bottom=0)
        
        # Validation reward comparison
        for i, (exp_name, exp_data) in enumerate(data.items()):
            val_df = exp_data['val']
            ax2.plot(val_df['step'], val_df['val_reward'], 
                    label=f'{exp_name} (Val)', 
                    color=self.colors[i],
                    linewidth=2,
                    alpha=0.8)
        
        ax2.set_title('Validation Reward Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Validation Reward', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined reward charts saved to: {save_path}")
        plt.show()
    
    def plot_training_metrics(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                            save_path: str = "grpo_training_metrics.png"):
        """Plot other metrics during training"""
        
        metrics = [
            ('train_loss', 'Training Loss'),
            ('gradient_norm', 'Gradient Norm'),
            ('token_entropy', 'Token Entropy'),
            ('learning_rate', 'Learning Rate')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx]
            
            for i, (exp_name, exp_data) in enumerate(data.items()):
                train_df = exp_data['train']
                
                if metric in train_df.columns:
                    ax.plot(train_df['step'], train_df[metric], 
                           label=exp_name, 
                           color=self.colors[i],
                           linewidth=2,
                           alpha=0.8)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Steps', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Use log scale for learning rate
            if metric == 'learning_rate':
                ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics charts saved to: {save_path}")
        plt.show()
    
    def generate_summary_table(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                             save_path: str = "grpo_experiment_summary.csv"):
        """Generate experiment summary table"""
        
        summary_data = []
        
        for exp_name, exp_data in data.items():
            train_df = exp_data['train']
            val_df = exp_data['val']
            
            # Calculate final values
            final_train_reward = train_df['train_reward'].iloc[-1] if len(train_df) > 0 else 0
            final_train_format = train_df['train_format_reward'].iloc[-1] if len(train_df) > 0 else 0
            final_train_answer = train_df['train_answer_reward'].iloc[-1] if len(train_df) > 0 else 0
            final_val_reward = val_df['val_reward'].iloc[-1] if len(val_df) > 0 else 0
            final_val_format = val_df['val_format_reward'].iloc[-1] if len(val_df) > 0 else 0
            final_val_answer = val_df['val_answer_reward'].iloc[-1] if len(val_df) > 0 else 0
            
            # Calculate maximum values
            max_train_reward = train_df['train_reward'].max() if len(train_df) > 0 else 0
            max_val_reward = val_df['val_reward'].max() if len(val_df) > 0 else 0
            
            summary_data.append({
                'Experiment Name': exp_name,
                'Final Training Reward': final_train_reward,
                'Final Training Format Reward': final_train_format,
                'Final Training Answer Reward': final_train_answer,
                'Final Validation Reward': final_val_reward,
                'Final Validation Format Reward': final_val_format,
                'Final Validation Answer Reward': final_val_answer,
                'Max Training Reward': max_train_reward,
                'Max Validation Reward': max_val_reward,
                'Training Steps': len(train_df),
                'Validation Count': len(val_df)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"Experiment summary table saved to: {save_path}")
        
        # Print summary table
        print("\n=== Experiment Summary ===")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def run_all_plots(self, output_dir: str = "plots"):
        """Run all plotting functions"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data = self.load_data()
        
        if not data:
            print("No experiment data loaded successfully!")
            return
        
        print(f"Successfully loaded data for {len(data)} experiments")
        
        # Generate all charts
        self.plot_metrics_comparison(data, os.path.join(output_dir, "metrics_comparison.png"))
        self.plot_combined_rewards(data, os.path.join(output_dir, "combined_rewards.png"))
        self.plot_training_metrics(data, os.path.join(output_dir, "training_metrics.png"))
        self.generate_summary_table(data, os.path.join(output_dir, "experiment_summary.csv"))
        
        print(f"\nAll charts saved to {output_dir} directory")

def polt_lr():
    # Define experiment list
    # Format: (experiment_name, training_log_file_path, validation_log_file_path)
    experiments = [
        ("LR 1e-5", 
         "training_logs/training_log_20250804_023101.csv",
         "training_logs/validation_log_20250804_023101.csv"),
        ("LR 2e-5", 
         "training_logs/training_log_20250804_032115.csv",
         "training_logs/validation_log_20250804_032115.csv"),
        ("LR 5e-5", 
         "training_logs/training_log_20250804_034855.csv",
         "training_logs/validation_log_20250804_034855.csv"),
    ]
    
    # Create plotter and run
    plotter = GRPOExperimentPlotter(experiments)
    plotter.run_all_plots("plots/lr")

def plot_baseline():
    experiments = [
        ("reinforce_with_baseline", 
         "training_logs/training_log_20250804_034855.csv",
         "training_logs/validation_log_20250804_034855.csv"),
        ("no_baseline", 
         "training_logs/training_log_20250804_043942.csv",
         "training_logs/validation_log_20250804_043942.csv"),
    ]
    
    # Create plotter and run
    plotter = GRPOExperimentPlotter(experiments)
    plotter.run_all_plots("plots/baseline")

def plot_normalize():
    experiments = [
        ("masked_normalize", 
         "training_logs/training_log_20250804_034855.csv",
         "training_logs/validation_log_20250804_034855.csv"),
        ("masked_mean", 
         "training_logs/training_log_20250804_053846.csv",
         "training_logs/validation_log_20250804_053846.csv"),
    ]
    
    # Create plotter and run
    plotter = GRPOExperimentPlotter(experiments)
    plotter.run_all_plots("plots/normalize")

def plot_std_normalize():
    experiments = [
        ("use_std_normalize", 
         "training_logs/training_log_20250804_034855.csv",
         "training_logs/validation_log_20250804_034855.csv"),
         ("no_std_normalize", 
         "training_logs/training_log_20250804_062149.csv",
         "training_logs/validation_log_20250804_062149.csv"),
    ]
    
    # Create plotter and run
    plotter = GRPOExperimentPlotter(experiments)
    plotter.run_all_plots("plots/std_normalize")

def plot_grpo():
    experiments = [
        # rollout_batch_size, epoch_per_rollout_batch, train_batch_size
        ("onpolicy_256_1_256", 
         "training_logs/training_log_20250804_062149.csv",
         "training_logs/validation_log_20250804_062149.csv"),
        ("grpo_offpolicy_256_2_512", 
         "training_logs/training_log_20250805_075341.csv",
         "training_logs/validation_log_20250805_075341.csv"),
        ("grpo_offpolicy_256_2_265 lr=2e-5", 
         "training_logs/training_log_20250805_091738.csv",
         "training_logs/validation_log_20250805_091738.csv"),
    ]
    
    # Create plotter and run
    plotter = GRPOExperimentPlotter(experiments)
    plotter.run_all_plots("plots/grpo_policy")

def plot_grpo_clip():
    experiments = [
        ("grpo_offpolicy_256_2_512", 
         "training_logs/training_log_20250805_075341.csv",
         "training_logs/validation_log_20250805_075341.csv"),
        ("grpo_offpolicy_256_2_512_no_clip", 
         "training_logs/training_log_20250805_100130.csv",
         "training_logs/validation_log_20250805_100130.csv"),
    ]

    plotter = GRPOExperimentPlotter(experiments)
    plotter.run_all_plots("plots/grpo_clip")

def plot_grpo_prompt():
    experiments = [
        ("grpo_offpolicy_256_2_512", 
         "training_logs/training_log_20250805_075341.csv",
         "training_logs/validation_log_20250805_075341.csv"),
        ("grpo_offpolicy_256_2_512_no_prompt", 
         "training_logs/training_log_20250805_140202.csv",
         "training_logs/validation_log_20250805_140202.csv"),
    ]

    plotter = GRPOExperimentPlotter(experiments)
    plotter.run_all_plots("plots/grpo_prompt")


def main():
    """Main function - define experiment configuration and run plotting"""
    # plot_lr()
    # plot_baseline()
    # plot_normalize()
    # plot_std_normalize()
    # plot_grpo()
    # plot_grpo_clip()
    plot_grpo_prompt()

if __name__ == "__main__":
    main() 