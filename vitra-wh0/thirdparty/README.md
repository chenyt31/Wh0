# Third-party Dependencies

Clone external repositories here when running deployment or low-level robot tools.

```bash
cd vitra-wh0/thirdparty
git clone https://github.com/unitreerobotics/xr_teleoperate.git
```

`deployment/client.py` imports Unitree G1, Inspire hand, image client, and XR control utilities from `xr_teleoperate`. Set `XR_TELEOPERATE_ROOT` if the checkout lives elsewhere.
