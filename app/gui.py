from typing import Callable, Optional

import webview


def launch_gui(url: str, on_closed: Optional[Callable[[], None]] = None) -> None:
    """
    Open the desktop GUI window pointed at the given URL and block until
    the user closes it. If `on_closed` is provided, it is invoked when the
    window is closed so the caller can shut the backend down cleanly.
    """
    window = webview.create_window(
        title="RAG Coding Assistant",
        url=url,
        width=1080,
        height=700,
        min_size=(720, 480),
        frameless=False,
        easy_drag=False,
    )
    if on_closed is not None:
        window.events.closed += on_closed
    webview.start(debug=False)


if __name__ == "__main__":
    launch_gui(url="http://127.0.0.1:5000/ui")
