"""
Command-line interface for stash-phash.

Usage:
    stash-phash video.mp4                    # Generate phash for a video
    stash-phash video1.mp4 video2.mp4       # Multiple files
    stash-phash --format int video.mp4       # Output as integer
    stash-phash --format hex video.mp4       # Output as hex (default)
    stash-phash -v video.mp4                 # Verbose output
    stash-phash --ffmpeg /path/to/ffmpeg video.mp4  # Custom ffmpeg path
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from stash_phash.utils import phash_to_signed, phash_to_string
from stash_phash.video import generate_video_phash


@click.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["hex", "int", "int-signed"]),
    default="hex",
    help="Output format: hex (default), int (unsigned), int-signed (signed, as stored in Stash DB)",
)
@click.option(
    "--ffmpeg",
    "ffmpeg_path",
    type=click.Path(),
    default=None,
    help="Path to ffmpeg executable (default: use system PATH)",
)
@click.option(
    "--ffprobe",
    "ffprobe_path",
    type=click.Path(),
    default=None,
    help="Path to ffprobe executable (default: use system PATH)",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose output")
@click.option(
    "-q", "--quiet", is_flag=True, default=False, help="Only output the hash values, no filenames"
)
@click.version_option(package_name="stash-phash")
def main(
    files: tuple[str, ...],
    output_format: str,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Generate Stash-compatible perceptual hashes for video files.

    FILES: One or more video files to process.

    Examples:

        stash-phash video.mp4

        stash-phash --format int video.mp4

        stash-phash -q *.mp4 > hashes.txt
    """
    errors = False

    for file_path in files:
        path = Path(file_path)

        if verbose:
            click.echo(f"Processing: {path}", err=True)

        try:
            # Generate phash
            phash = generate_video_phash(path, ffmpeg_path, ffprobe_path)

            # Format output
            if output_format == "hex":
                hash_str = phash_to_string(phash)
            elif output_format == "int":
                hash_str = str(phash)
            else:  # int-signed
                hash_str = str(phash_to_signed(phash))

            # Print result
            if quiet:
                click.echo(hash_str)
            else:
                click.echo(f"{path}: {hash_str}")

        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            errors = True
        except Exception as e:
            click.echo(f"Error processing {path}: {e}", err=True)
            if verbose:
                import traceback

                traceback.print_exc()
            errors = True

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
