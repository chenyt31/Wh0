from typing import Dict, List, Tuple

HAND_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === ego4d + egoexo4d + ssv2 + epic Dataset ===
    "magic_mix": [
        ("ego4d_cooking_and_cleaning", 1.0),
        ("egoexo4d", 3.0),
        ("epic", 1.0),
        ('ssv2', 5.0),
        ('ego4d_other', 0.5)
    ],
    "magic_mix_cooking_and_cleaning": [
        ("ego4d_cooking_and_cleaning", 1.0),
        ("egoexo4d", 3.0),
        ("epic", 1.0),
        ('ssv2', 5.0),
    ],
    "real_only": [
        ("g1_dataset", 1.0)
    ],
    "WM-H_50k" : [
        ("WM-H", 1.0),
        # ("g1_dataset", 0.02)
        ("g1_dataset", 0.4)
    ]
}
