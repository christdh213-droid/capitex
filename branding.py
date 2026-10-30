"""Local-first CAPITEX branding and safe circular asset images."""
import base64
from functools import lru_cache
from html import escape
from pathlib import Path
import requests
import config


@lru_cache(maxsize=128)
def asset_logo_source(ticker: str, try_remote: bool = True) -> str:
    """Resolve remote logos on the server so broken images never reach the UI.

    SVG badges are the offline default; publisher images are rendered inside a
    circular frame. URLs are a curated allowlist, never user-supplied addresses.
    """
    asset = config.ASSET_UNIVERSE.get(ticker.upper())
    if not asset:
        return _data_uri(config.LOGO_PATH)
    if try_remote:
        try:
            response = requests.get(asset['logo_url'], timeout=3)
            response.raise_for_status()
            mime = response.headers.get('Content-Type', '').split(';')[0].lower()
            raw = response.content
            # Only supported, recognisable raster payloads enter the browser.
            valid = ((mime == 'image/png' and raw.startswith(b'\x89PNG\r\n\x1a\n')) or
                     (mime == 'image/jpeg' and raw.startswith(b'\xff\xd8\xff')) or
                     (mime in ('image/x-icon', 'image/vnd.microsoft.icon') and raw.startswith(b'\x00\x00\x01\x00')))
            if valid and len(raw) <= 2_000_000:
                return f'data:{mime};base64,' + base64.b64encode(raw).decode('ascii')
        except requests.RequestException:
            pass
    return _data_uri(Path(asset['logo_fallback']))


def _data_uri(path: Path) -> str:
    return 'data:image/svg+xml;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def asset_logo_html(ticker: str, size: int = 48, try_remote: bool = True) -> str:
    name = config.ASSET_UNIVERSE.get(ticker.upper(), {}).get('name', ticker)
    source = asset_logo_source(ticker, try_remote)
    return (f'<img src="{source}" alt="{escape(name, quote=True)}" width="{int(size)}" '
            f'height="{int(size)}" style="border-radius:50%;object-fit:contain;'
            'background:#181820;padding:5px;box-sizing:border-box"/>')
