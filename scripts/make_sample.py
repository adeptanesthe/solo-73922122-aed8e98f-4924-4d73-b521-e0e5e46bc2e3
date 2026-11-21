"""Generate a synthetic fault screenshot and matching log snippet for manual testing.

Usage:
    python scripts/make_sample.py --out samples
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def build_screenshot(path: Path, width: int = 960, height: int = 540) -> None:
    """Draw a synthetic 'web app error' screenshot: red banner + dialog + log-ish text."""
    img = Image.new("RGB", (width, height), (245, 246, 248))
    d = ImageDraw.Draw(img)

    # Browser chrome
    d.rectangle([0, 0, width, 36], fill=(52, 54, 58))
    d.ellipse([12, 12, 24, 24], fill=(255, 95, 86))
    d.ellipse([32, 12, 44, 24], fill=(255, 189, 46))
    d.ellipse([52, 12, 64, 24], fill=(39, 201, 63))
    d.rectangle([80, 8, width - 20, 28], fill=(70, 72, 76))
    d.text((90, 11), "https://shop.example.internal/checkout", fill=(220, 220, 220))

    # Red error banner (key region #1)
    d.rectangle([0, 60, width, 120], fill=(198, 40, 40))
    d.text((24, 78), "ERROR 502 - Bad Gateway: upstream service unavailable", fill=(255, 255, 255))

    # Amber warning strip (key region #2)
    d.rectangle([0, 130, width, 158], fill=(255, 171, 0))
    d.text((24, 136), "Warning: payment-gateway latency above 5000ms threshold", fill=(40, 40, 40))

    # Modal dialog (key region #3)
    d.rectangle([280, 200, 680, 380], fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    d.rectangle([280, 200, 680, 232], fill=(58, 58, 60))
    d.text((296, 208), "Checkout failed", fill=(255, 255, 255))
    d.text((300, 250), "The payment service did not respond in time.", fill=(30, 30, 30))
    d.text((300, 274), "Request ID: req-7f3a2c  (timeout after 5000 ms)", fill=(90, 90, 90))
    d.rectangle([300, 320, 420, 352], fill=(198, 40, 40))
    d.text((330, 330), "Retry", fill=(255, 255, 255))
    d.rectangle([440, 320, 560, 352], fill=(224, 224, 224))
    d.text((478, 330), "Cancel", fill=(40, 40, 40))

    # Body text lines
    for i, line in enumerate(
        [
            "Your order could not be completed.",
            "If the problem persists, contact support with the request ID above.",
            "Status page: https://status.example.internal",
        ]
    ):
        d.text((40, 420 + i * 24), line, fill=(60, 60, 60))

    img.save(path)


def build_logs(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2026-11-18T10:15:01Z INFO  checkout-api request received user=u-4821 cart=c-991",
                "2026-11-18T10:15:02Z INFO  checkout-api calling payment-gateway /v1/charge",
                "2026-11-18T10:15:07Z ERROR payment-gateway client timeout after 5000ms req=req-7f3a2c",
                "2026-11-18T10:15:07Z WARN  checkout-api upstream payment-gateway unhealthy, circuit=open",
                "2026-11-18T10:15:07Z ERROR checkout-api charge failed: upstream connect timeout req=req-7f3a2c",
                "2026-11-18T10:15:07Z INFO  checkout-api responded 502 to client req=req-7f3a2c",
                "2026-11-18T10:15:09Z INFO  healthz ok db=up cache=up",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build_screenshot(args.out / "fault_screenshot.png")
    build_logs(args.out / "app_logs.txt")
    (args.out / "question.txt").write_text("Checkout page shows 502 and payments time out. What is the likely cause?\n")
    print(f"wrote samples to {args.out.resolve()}")


if __name__ == "__main__":
    main()
