"""Launch a browser viewer with live, native Scaffold-GS rendering."""

from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", required=True, help="Training output directory containing cfg_args")
    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration; -1 selects latest")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max_resolution", type=int, default=1600, help="Longest image side while stationary")
    parser.add_argument("--moving_resolution", type=int, default=800, help="Longest image side while moving")
    parser.add_argument("--fps", type=int, default=15, help="Maximum render rate per browser client")
    args = parser.parse_args()
    if not 128 <= args.moving_resolution <= args.max_resolution <= 4096 or not 1 <= args.fps <= 60:
        parser.error("Require 128 <= moving_resolution <= max_resolution <= 4096 and 1 <= fps <= 60")
    try:
        from utils.viewer_app import ScaffoldViewer
    except ImportError as error:
        parser.exit(1, f"Viewer dependencies unavailable: {error}\nRun: pip install -r requirements-viewer.txt\n")
    from utils.viewer_renderer import ScaffoldRenderer

    try:
        renderer = ScaffoldRenderer(args.model_path, args.iteration, args.gpu)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Cannot start native viewer: {error}\n")
    viewer = ScaffoldViewer(renderer, host=args.host, port=args.port, max_resolution=args.max_resolution,
                            moving_resolution=args.moving_resolution, fps=args.fps)
    print(f"Loaded iteration {renderer.iteration}: {renderer.anchor_count:,} anchors")
    print(f"Open http://localhost:{args.port} (or http://<server-ip>:{args.port} from another computer)")
    print("Live Scaffold-GS inference runs on this CUDA machine. Press Ctrl+C to stop.")
    try:
        viewer.run()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()


if __name__ == "__main__":
    main()
