# Demo script for visualizing hand VLA episodes.
import sys
import os

# Adjust system path to include the vitra root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
vitra_root = os.path.dirname(current_dir)
project_root = os.path.dirname(vitra_root)
for path in (vitra_root, project_root):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, vitra_root)
sys.path.insert(0, project_root)

import argparse

from visualization.visualize_core import Config, HandVisualizer

# --- Main Execution Function ---
def main():
    """Main execution function, including argument parsing."""
    parser = argparse.ArgumentParser(description="Visualize hand VLA episodes with customizable paths.")
    
    # Path Arguments
    parser.add_argument(
        '--video_root',
        type=str,
        default='../assets/debug_eval/wm-h/videos',
        help='Root directory containing the video files.'
    )
    parser.add_argument(
        '--label_root',
        type=str,
        default='../assets/debug_eval/wm-h/annotations',
        help='Root directory containing the episode label (.npy) files.'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='../output_videos/visualize',
        help='Directory to save the output visualization videos.'
    )
    parser.add_argument(
        '--mano_model_path',
        type=str,
        default='./weights/mano',
        help='Path to the MANO model files.'
    )
    
    # Visualization Arguments
    parser.add_argument(
        '--render_gradual_traj',
        action='store_true',
        help='Set flag to render a gradual trajectory (full mode).'
    )
    parser.add_argument(
        '--modes',
        type=str,
        default='cam,first',
        help='Comma-separated render modes, for example: cam or cam,first.'
    )
    parser.add_argument(
        '--hide_left',
        action='store_true',
        help='Do not render left-hand labels.'
    )
    parser.add_argument(
        '--hide_right',
        action='store_true',
        help='Do not render right-hand labels.'
    )
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=0,
        help='Maximum number of episodes to render. 0 renders all episodes.'
    )
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Skip MANO/PyTorch3D mesh rendering and save captioned video previews.'
    )

    args = parser.parse_args()

    # 1. Initialize Visualizer with parsed arguments
    config = Config(args)
    
    # Ensure save path exists
    os.makedirs(config.SAVE_PATH, exist_ok=True)
    
    visualizer = HandVisualizer(config, render_gradual_traj=args.render_gradual_traj)

    # 2. Load Episode Names
    try:
        all_episode_names_npy = sorted(name for name in os.listdir(args.label_root) if name.endswith('.npy'))
        all_episode_names = [n.split('.npy')[0] for n in all_episode_names_npy]
        if args.max_episodes > 0:
            all_episode_names = all_episode_names[:args.max_episodes]

    except FileNotFoundError:
        print(f"Error: Episode list directory not found at {args.label_root}. Cannot proceed.")
        return

    # 3. Process All Episodes
    print(f"--- Running Hand Visualizer ---")
    print(f"Video Root: {config.VIDEO_ROOT}")
    print(f"Label Root: {config.LABEL_ROOT}")
    print(f"Save Path: {config.SAVE_PATH}")
    print(f"MANO Model Path: {config.MANO_MODEL_PATH}")
    print(f"Rendering Gradual Trajectory: {args.render_gradual_traj}")
    print(f"Mesh Rendering: {not args.no_render}")
    print(f"-------------------------------")

    for episode_name in all_episode_names:
        visualizer.process_episode(episode_name)


if __name__ == '__main__':
    main()
