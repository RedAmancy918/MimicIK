# MimicIK project website

The project homepage is hosted at <https://redamancy918.github.io/MimicIK/>.
This directory contains buildless HTML, CSS, JavaScript, and web-ready media.
GitHub Pages publishes these files through the repository's Pages workflow.

## Editing and local preview

Edit `index.html` for content, `style.css` for appearance, and `app.js` for
interactions. Keep asset references relative so the page works under the
`/MimicIK/` project path. No JavaScript package installation or build is needed.

From the repository root:

```bash
python3 -m http.server 8768 --directory website --bind 127.0.0.1
```

Open `http://127.0.0.1:8768/`. Reload after editing. Check narrow and desktop
layouts, video controls, demonstration tabs, and copy buttons before pushing.

## Demonstration media

These author-provided recordings show earlier MimicIK hardware experiments.
The downloadable AIRBOT checkpoints come from a later controlled training
series. The excerpts retain their original playback speed and contain no
generated motion. They were converted to silent 720p H.264, SDR BT.709 with
fast-start metadata for browser playback; the JPEG posters come from the clips.

- `assets/cup-placement.mp4`: `mipfk_pi.MOV`, 28–52 seconds.
- `assets/bottle-pickup.mp4`: `mipfk_pi_pickbottle.MOV`, 3.5–34 seconds.
- `assets/teleoperation.mp4`: `mip_ood.MOV`, 9.5–33.5 seconds.
- `assets/hero.mp4`: `mipfk_pi.MOV`, 28–40 seconds.

The repository's MIT code license does not automatically license the
experimental footage. No additional public media license is asserted here.
