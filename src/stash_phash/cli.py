"""
Command-line interface for stash-phash.

Usage:
    stash-phash video.mp4                    # Generate phash for a video
    stash-phash video1.mp4 video2.mp4       # Multiple files
    stash-phash https://example.com/vid.mp4  # Remote URL
    stash-phash https://example.com/stream.m3u8  # HLS stream
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
from stash_phash.video import STREAM_TIMEOUT, generate_video_phash, is_url


@click.command()
@click.argument("sources", nargs=-1, required=True)
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
@click.option(
    "--timeout",
    "timeout",
    type=int,
    default=None,
    help=f"Timeout per frame extraction in seconds (default: {STREAM_TIMEOUT}s for streams)",
)
@click.version_option(package_name="stash-phash")
def main(
    sources: tuple[str, ...],
    output_format: str,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
    verbose: bool,
    quiet: bool,
    timeout: int | None,
) -> None:
    """Generate Stash-compatible perceptual hashes for video files and streams.

    SOURCES: One or more video files or URLs to process.

    Supports local files and remote streams (HTTP, HTTPS, HLS, RTMP, etc.)

    Examples:

    \b
        # Local file
        stash-phash video.mp4

    \b
        # Remote URL
        stash-phash https://example.com/video.mp4

    \b
        # HLS stream
        stash-phash https://example.com/stream.m3u8

    \b
        # Multiple sources
        stash-phash video.mp4 https://example.com/video.mp4

    \b
        # Output as integer for database storage
        stash-phash --format int-signed video.mp4
    """
    errors = False

    for source in sources:
        # Determine display name
        display_name = source if is_url(source) else str(Path(source))

        if verbose:
            click.echo(f"Processing: {display_name}", err=True)

        try:
            # Validate local files exist before processing
            if not is_url(source):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Video file not found: {path}")

            # Progress callback for verbose mode
            def progress(frame: int, total: int) -> None:
                if verbose:
                    click.echo(f"  Extracting frame {frame}/{total}...", err=True)

            # Generate phash
            phash = generate_video_phash(
                source,
                ffmpeg_path,
                ffprobe_path,
                timeout=timeout,
                progress_callback=progress if verbose else None,
            )

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
                click.echo(f"{display_name}: {hash_str}")

        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            errors = True
        except Exception as e:
            click.echo(f"Error processing {display_name}: {e}", err=True)
            if verbose:
                import traceback

                traceback.print_exc()
            errors = True

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
